
from typing import Literal
from langchain_core.pydantic_v1 import BaseModel, Field

class QueryClassification(BaseModel):
    """Structured output for the Instructor classification step."""
    category: Literal['bank account','credit card','kyc', 'fraud', 'dispute', 'general', 'loan', 'others']
    is_banking_related: bool
    refined_query: str = Field(
        ...,
        description="Refine/rewrite the user's query for downstream retrieval or tool calls."
    )
