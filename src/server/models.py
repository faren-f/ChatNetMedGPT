from datetime import datetime
from enum import Enum
from typing import List, Optional, Annotated

from fastapi import Body
from pydantic import BaseModel


class ChatRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class DrugResponseDTO(BaseModel):
    predictions: List[str]
    prediction_type: str
    message: str


class ChatMessage(BaseModel):
    id: str
    role: ChatRole
    text: str
    createdAt: Annotated[datetime, Body()]
    pending: Optional[bool] = None
    error: Optional[bool] = None
    drugResponse: Optional[DrugResponseDTO] = None


class ChatRequest(BaseModel):
    history: List[ChatMessage]
    message: str