import base64
import json
import os
from typing import Optional, Dict
import logging
from datetime import datetime
import uvicorn
from anthropic import Anthropic
from fastapi import FastAPI, UploadFile, Form, HTTPException
import httpx


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HomeBox LLM Image Classifier",
    description="Classify household items using Claude Vision and add them to HomeBox",
    version="1.0.0"
)

# Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
HOMEBOX_URL = os.getenv("HOMEBOX_URL", "http://localhost:7745")
HOMEBOX_USERNAME = os.getenv("HOMEBOX_USERNAME")
HOMEBOX_PASSWORD = os.getenv("HOMEBOX_PASSWORD")

# Initialize Anthropic client
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

# Common household item categories for labels
COMMON_CATEGORIES = [
    "Clothing", "Electronics", "Books", "Kitchen", "Furniture",
    "Decor", "Tools", "Documents", "Toys", "Sports", "Garden",
    "Bathroom", "Office", "Cleaning", "Misc"
]


class HomeBoxAPI:
    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url.rstrip('/')
        self.username = username
        self.password = password
        self.token = None
        self.token_expires = None
        self.client = httpx.AsyncClient(timeout=30.0)

    async def _ensure_authenticated(self):
        """Ensure we have a valid authentication token"""
        if not self.token or (self.token_expires and datetime.now() > self.token_expires):
            await self._login()

    async def _login(self):
        """Login to HomeBox and store the bearer token"""
        try:
            response = await self.client.post(
                f"{self.base_url}/api/v1/users/login",
                json={
                    "username": self.username,
                    "password": self.password
                }
            )
            response.raise_for_status()

            data = response.json()
            self.token = data["token"]
            # Parse expiration if provided
            if "expiresAt" in data:
                # Handle expiration logic here if needed
                pass

            logger.info("Successfully authenticated with HomeBox")

        except Exception as e:
            logger.error(f"Failed to authenticate with HomeBox: {e}")
            raise HTTPException(
                status_code=401, detail="Failed to authenticate with HomeBox")

    async def _make_request(self, method: str, endpoint: str, **kwargs):
        """Make authenticated request to HomeBox API"""
        await self._ensure_authenticated()

        headers = kwargs.get("headers", {})
        headers["Authorization"] = f"Bearer {self.token}"
        kwargs["headers"] = headers

        url = f"{self.base_url}/api{endpoint}"
        response = await self.client.request(method, url, **kwargs)

        if response.status_code == 401:
            # Token might be expired, try to re-authenticate
            await self._login()
            headers["Authorization"] = f"Bearer {self.token}"
            response = await self.client.request(method, url, **kwargs)

        response.raise_for_status()
        return response

    async def get_locations(self) -> list:
        """Get all locations from HomeBox"""
        response = await self._make_request("GET", "/v1/locations")
        return response.json()

    async def find_location_by_name(self, name: str) -> Optional[Dict]:
        """Find location by name (case-insensitive)"""
        locations = await self.get_locations()
        name_lower = name.lower().strip()

        for location in locations:
            if location["name"].lower() == name_lower:
                return location
        return None

    async def create_location(self, name: str, description: str = "") -> Dict:
        """Create a new location"""
        response = await self._make_request(
            "POST",
            "/v1/locations",
            json={
                "name": name,
                "description": description
            }
        )
        return response.json()

    async def find_or_create_location(self, location_name: str) -> str:
        """Find existing location or create new one, return location ID"""
        # Try to find existing location
        existing = await self.find_location_by_name(location_name)
        if existing:
            logger.info(f"Found existing location: {existing['name']}")
            return existing["id"]

        # Create new location
        logger.info(f"Creating new location: {location_name}")
        new_location = await self.create_location(
            name=location_name,
            description=f"Auto-created location for {location_name}"
        )
        return new_location["id"]

    async def get_labels(self) -> list:
        """Get all labels from HomeBox"""
        response = await self._make_request("GET", "/v1/labels")
        return response.json()

    async def find_label_by_name(self, name: str) -> Optional[Dict]:
        """Find label by name (case-insensitive)"""
        labels = await self.get_labels()
        name_lower = name.lower().strip()

        for label in labels:
            if label["name"].lower() == name_lower:
                return label
        return None

    async def create_label(self, name: str, description: str = "", color: str = "") -> Dict:
        """Create a new label"""
        response = await self._make_request(
            "POST",
            "/v1/labels",
            json={
                "name": name,
                "description": description,
                "color": color
            }
        )
        return response.json()

    async def find_or_create_label(self, label_name: str) -> str:
        """Find existing label or create new one, return label ID"""
        existing = await self.find_label_by_name(label_name)
        if existing:
            return existing["id"]

        # Create new label
        logger.info(f"Creating new label: {label_name}")
        new_label = await self.create_label(
            name=label_name,
            description=f"Auto-created category for {label_name}"
        )
        return new_label["id"]

    async def create_item(self, name: str, description: str, location_id: str, label_ids: list = None) -> Dict:
        """Create a new item in HomeBox"""
        item_data = {
            "name": name,
            "description": description,
            "locationId": location_id,
            "quantity": 1
        }

        if label_ids:
            item_data["labelIds"] = label_ids

        response = await self._make_request(
            "POST",
            "/v1/items",
            json=item_data
        )
        return response.json()

    async def upload_image_attachment(self, item_id: str, image_data: bytes, filename: str) -> Dict:
        """Upload image as attachment to an item"""
        files = {
            "file": (filename, image_data, "image/jpeg"),
            "type": (None, "photo"),
            "name": (None, filename),
            "primary": (None, "true")
        }

        response = await self._make_request(
            "POST",
            f"/v1/items/{item_id}/attachments",
            files=files
        )
        return response.json()


# Initialize HomeBox API client
homebox = HomeBoxAPI(HOMEBOX_URL, HOMEBOX_USERNAME, HOMEBOX_PASSWORD)


async def classify_image_with_claude(image_data: bytes) -> Dict[str, str]:
    """Use Claude 4 Sonnet to classify the household item in the image"""

    # Convert image to base64
    image_b64 = base64.b64encode(image_data).decode()

    # Specialized prompt for household item classification
    prompt = f"""Analyze this image of a household item. You are helping someone organize their home inventory during packing/moving.

Return a JSON object with exactly these fields:
- "name": A clear, descriptive name for the item (e.g., "Black leather office chair", "Stainless steel coffee maker", "Red wool sweater")
- "category": Choose the BEST category from: {', '.join(COMMON_CATEGORIES)}
- "description": A brief 1-2 sentence description focusing on key identifying features

Guidelines:
- Be specific but concise with the name
- Focus on items that would typically be inventoried in a home
- If multiple items are visible, focus on the main/central item
- Use practical, everyday language someone would use when looking for the item

Return only valid JSON, no other text."""

    try:
        message = anthropic_client.messages.create(
            # Using Claude 3.5 Sonnet (current best vision model)
            model="claude-3-5-sonnet-20241022",
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )

        # Parse the JSON response
        response_text = message.content[0].text.strip()
        logger.info(f"Claude response: {response_text}")

        # Try to extract JSON if there's extra text
        if response_text.startswith('```json'):
            response_text = response_text.split(
                '```json')[1].split('```')[0].strip()
        elif response_text.startswith('```'):
            response_text = response_text.split(
                '```')[1].split('```')[0].strip()

        result = json.loads(response_text)

        # Validate required fields
        required_fields = ["name", "category", "description"]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")

        # Validate category is in our list
        if result["category"] not in COMMON_CATEGORIES:
            logger.warning(
                f"Unusual category: {result['category']}, defaulting to 'Misc'")
            result["category"] = "Misc"

        return result

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Claude response as JSON: {e}")
        logger.error(f"Raw response: {response_text}")
        raise HTTPException(
            status_code=500, detail="Failed to parse AI classification response")

    except Exception as e:
        logger.error(f"Error calling Claude API: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to classify image with AI")


@app.post("/classify-item")
async def classify_item(
    image: UploadFile,
    location: str = Form(...,
                         description="Location name (e.g., 'large-box-1', 'kitchen-cabinet')")
):
    """
    Classify a household item image and add it to HomeBox inventory

    - **image**: Image file of the household item
    - **location**: Location name where the item is stored
    """

    try:
        # Validate image
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, detail="Please upload a valid image file")

        # Read image data
        image_data = await image.read()
        if len(image_data) == 0:
            raise HTTPException(status_code=400, detail="Image file is empty")

        logger.info(
            f"Processing image: {image.filename}, size: {len(image_data)} bytes, location: {location}")

        # Step 1: Classify image with Claude
        classification = await classify_image_with_claude(image_data)
        logger.info(f"Classification result: {classification}")

        # Step 2: Find or create location
        location_id = await homebox.find_or_create_location(location)

        # Step 3: Find or create label for category
        label_id = await homebox.find_or_create_label(classification["category"])

        # Step 4: Create item in HomeBox
        item = await homebox.create_item(
            name=classification["name"],
            description=classification["description"],
            location_id=location_id,
            label_ids=[label_id]
        )

        logger.info(f"Created item with ID: {item['id']}")

        # Step 5: Upload image as attachment
        attachment_result = await homebox.upload_image_attachment(
            item_id=item["id"],
            image_data=image_data,
            filename=image.filename or "item_photo.jpg"
        )

        logger.info("Successfully uploaded image attachment")

        return {
            "success": True,
            "item": {
                "id": item["id"],
                "name": item["name"],
                "description": classification["description"],
                "category": classification["category"],
                "location": location,
                "assetId": item.get("assetId")
            },
            "message": f"Successfully added '{classification['name']}' to HomeBox inventory"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing item: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process item: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "homebox_url": HOMEBOX_URL
    }


@app.get("/")
async def root():
    """API information"""
    return {
        "title": "HomeBox LLM Image Classifier",
        "description": "Classify household items using Claude Vision and add them to HomeBox",
        "version": "1.0.0",
        "endpoints": {
            "classify": "/classify-item",
            "health": "/health",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
