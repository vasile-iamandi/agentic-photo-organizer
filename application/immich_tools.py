"""
Imich API Tools for LangChain
Uses the Imich REST API to search, upload, and manage assets and albums.
"""

import os
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

# Load environment variables
IMMICH_SERVER_URL = os.getenv("IMMICH_SERVER_URL", "http://localhost:2283")
IMMICH_API_KEY = os.getenv("IMMICH_API_KEY")


class Asset(BaseModel):
    """Model for Imich asset response."""
    id: str
    type: str
    originalPath: Optional[str] = None
    originalFileName: Optional[str] = None
    mimeType: Optional[str] = None
    createdAt: Optional[str] = None
    modifiedAt: Optional[str] = None

class SearchAssetsInput(BaseModel):
    """Input model for searching assets."""
    query: str = Field(description="Search query for assets (e.g., 'beach', '2024-01-*', 'cat')")
    page: int = Field(default=1, ge=1, description="Page number for pagination")
    size: int = Field(default=50, ge=1, le=100, description="Number of results per page")


class CreateAlbumInput(BaseModel):
    """Input model for creating an album."""
    album_name: str = Field(description="The name of the new album")
    description: Optional[str] = Field(default="", description="Optional description of the album")
    asset_ids: Optional[List[str]] = Field(default=None, description="List of asset IDs to include in the album. When creating an album to store specific photos, ALWAYS provide the found asset IDs here.")


class AddAssetsToAlbumInput(BaseModel):
    """Input model for adding assets to an existing album."""
    album_id: str = Field(description="The ID of the album")
    asset_ids: List[str] = Field(description="List of asset IDs to add to the album")

class ImichAPI:
    """Wrapper for Imich API calls."""
    
    def __init__(self, server_url: str = None, api_key: str = None):
        self.server_url = server_url or IMMICH_SERVER_URL
        self.api_key = api_key or IMMICH_API_KEY
        if not self.api_key:
            raise ValueError("IMMICH_API_KEY is required")
        self.client = httpx.Client()
        
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/json",
            "x-api-key": self.api_key,
        }
    
    def _get_url(self, endpoint: str) -> str:
        return f"{self.server_url.rstrip('/')}/api{endpoint}"
    
    def search_assets(
        self, 
        query: str = None, 
        page: int = 1, 
        size: int = 50,
        is_archived: bool = False,
        is_trashed: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search for assets in Imich.
        """
        endpoint = "/search/smart"

        payload = {
            "query": query or "*",
            "page": page,
            "size": size,
            "withDeleted": is_trashed,
        }

        url = self._get_url(endpoint)
        headers = self._get_headers()
        headers["Content-Type"] = "application/json"

        response = self.client.post(
            url, 
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()

        data = response.json()
        return data.get("assets", {}).get("items", [])

    def get_asset(self, asset_id: str) -> Dict[str, Any]:
        """Get a single asset by ID."""
        endpoint = f"/assets/{asset_id}"
        url = self._get_url(endpoint)
        response = self.client.get(
            url,
            headers=self._get_headers(),
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    def upload_asset(self, file_path: str) -> Dict[str, Any]:
        """Upload an asset to Imich."""
        import datetime
        stats = os.stat(file_path)
        endpoint = "/assets"
        url = self._get_url(endpoint)
        
        mtime = datetime.datetime.fromtimestamp(stats.st_mtime).isoformat() + "Z"

        data = {
            "fileCreatedAt": mtime,
            "fileModifiedAt": mtime,
            "isFavorite": "false",
        }

        with open(file_path, "rb") as f:
            files = {"assetData": (os.path.basename(file_path), f)}
            response = self.client.post(
                url,
                headers=self._get_headers(),
                data=data,
                files=files,
                timeout=60
            )
        response.raise_for_status()
        return response.json()

    def get_albums(self) -> List[Dict[str, Any]]:
        """List all albums."""
        endpoint = "/albums"
        url = self._get_url(endpoint)
        response = self.client.get(
            url,
            headers=self._get_headers(),
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    def create_album(
        self,
        album_name: str,
        description: str = "",
        asset_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create a new album with optional asset IDs."""
        endpoint = "/albums"
        url = self._get_url(endpoint)
        headers = self._get_headers()
        headers["Content-Type"] = "application/json"
        
        payload = {
            "albumName": album_name,
            "description": description or "",
            "assetIds": asset_ids or []
        }
        
        response = self.client.post(
            url,
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    def add_assets_to_album(self, album_id: str, asset_ids: List[str]) -> List[Dict[str, Any]]:
        """Add assets to an existing album."""
        endpoint = f"/albums/{album_id}/assets"
        url = self._get_url(endpoint)
        headers = self._get_headers()
        headers["Content-Type"] = "application/json"
        
        payload = {
            "ids": asset_ids
        }
        
        response = self.client.put(
            url,
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    def close(self):
        """Close the HTTP client."""
        self.client.close()

##################################################################
# Singleton instance
##################################################################
immich_client = ImichAPI()

##################################################################
# --- LangChain Tools ---
##################################################################
from langchain_core.tools import tool

@tool
async def search_immich_assets(
    query: str, 
    page: int = 1, 
    size: int = 50
) -> str:
    """
    Search for assets in your Immich server using Smart/CLIP Search.
    
    IMPORTANT: The Immich CLIP embedding model works best with English search queries (e.g. use 'dog', 'cat', 'beach', 'red car', 'sunset', 'forest').
    Always translate semantic search concepts from other languages to concise English keywords before calling this tool.
    
    Args:
        query: Search query in English for best accuracy (e.g., 'dog', 'beach', 'cat', 'mountain')
        page: Page number (default: 1)
        size: Number of results per page (default: 50, max: 100)
        
    Returns:
        JSON string with list of matching assets
    """
    try:
        assets = immich_client.search_assets(query=query, page=page, size=size)
        simplified = [
            {
                "id": a.get("id"),
                "originalFileName": a.get("originalFileName"),
                "fileCreatedAt": a.get("fileCreatedAt")
            }
            for a in assets
        ]
        return f"Found {len(simplified)} assets:\n{simplified}"
    except Exception as e:
        return f"Error searching assets: {str(e)}"

@tool
async def get_immich_asset(asset_id: str) -> str:
    """
    Get details for a specific asset from Imich.
    
    Args:
        asset_id: The ID of the asset to retrieve
        
    Returns:
        JSON string with asset details
    """
    try:
        asset = immich_client.get_asset(asset_id)
        return f"Asset details:\n{asset}"
    except Exception as e:
        return f"Error getting asset: {str(e)}"

@tool
async def create_immich_album(
    album_name: str,
    description: str = "",
    asset_ids: Optional[List[str]] = None
) -> str:
    """
    Create a new album on the Immich server, adding specific asset IDs to it.
    
    Args:
        album_name: The name of the album to create (e.g., 'Pisici', 'Vacanta 2024')
        description: Optional description for the album
        asset_ids: List of asset IDs to add immediately to the new album. If the user asked to put photos into the album, you MUST provide the found asset IDs here.
        
    Returns:
        JSON string with details of the created album
    """
    try:
        album = immich_client.create_album(album_name=album_name, description=description, asset_ids=asset_ids)
        return f"Album created successfully:\n{album}"
    except Exception as e:
        return f"Error creating album: {str(e)}"

@tool
async def get_immich_albums() -> str:
    """
    Get the list of existing albums on the Imich server.
    
    Returns:
        JSON string with list of albums
    """
    try:
        albums = immich_client.get_albums()
        return f"Found {len(albums)} albums:\n{albums}"
    except Exception as e:
        return f"Error fetching albums: {str(e)}"

@tool
async def add_assets_to_immich_album(
    album_id: str,
    asset_ids: List[str]
) -> str:
    """
    Add existing assets/photos to an existing album on the Imich server.
    
    Args:
        album_id: The ID of the target album
        asset_ids: The list of asset IDs to add into the album
        
    Returns:
        Result of adding assets to the album
    """
    try:
        result = immich_client.add_assets_to_album(album_id=album_id, asset_ids=asset_ids)
        return f"Assets added to album successfully:\n{result}"
    except Exception as e:
        return f"Error adding assets to album: {str(e)}"

##################################################################
# Tools list for LangChain
##################################################################
immich_tools = [
    search_immich_assets,
    get_immich_asset,
    create_immich_album,
    get_immich_albums,
    add_assets_to_immich_album
]
