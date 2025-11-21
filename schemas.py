"""
Database Schemas for FaceCare app

Each Pydantic model represents a collection in MongoDB. The collection name is the lowercase of the class name.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class Analysis(BaseModel):
    """
    Stores a single face analysis result for an uploaded image
    Collection name: "analysis"
    """
    image_url: str = Field(..., description="Public URL or path where the uploaded image is stored")
    detected_issues: List[str] = Field(default_factory=list, description="List of detected skin concerns")
    skin_type: Optional[str] = Field(None, description="Predicted skin type: dry, oily, combination, normal, sensitive")
    recommendations: List[str] = Field(default_factory=list, description="Actionable skincare recommendations")


class ProductSuggestion(BaseModel):
    """
    Stores suggested products mapped to issues
    Collection name: "productsuggestion"
    """
    issue: str = Field(..., description="Skin concern this product addresses")
    product_name: str = Field(..., description="Name of the product")
    category: str = Field(..., description="Category e.g., cleanser, moisturizer, SPF")
    ingredients: List[str] = Field(default_factory=list, description="Key ingredients")
    notes: Optional[str] = Field(None, description="Additional usage notes")
