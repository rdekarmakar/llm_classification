"""
Pydantic models for input validation and data structures.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from enum import Enum


class TicketCategory(str, Enum):
    """Ticket categories - fixed naming to match values."""
    CLAIM_DENIAL = "claim_denial"
    ACCOUNT_ACCESS = "account_access"
    COVERAGE_INQUIRY = "coverage_inquiry"
    DEPENDENT_COVERAGE_ISSUE = "dependent_coverage_issue"
    BILLING_ISSUE = "billing_issue"
    OTHER = "other"


class CustomerSentiment(str, Enum):
    """Customer sentiment levels."""
    ANGRY = "angry"
    FRUSTRATED = "frustrated"
    NEUTRAL = "neutral"
    SATISFIED = "satisfied"


class TicketUrgency(str, Enum):
    """Ticket urgency levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TicketClassification(BaseModel):
    """Structured classification result from LLM."""
    category: TicketCategory
    urgency: TicketUrgency
    sentiment: CustomerSentiment
    confidence: float = Field(ge=0, le=1, description="Confidence score for the classification")
    key_information: List[str] = Field(description="List of key points extracted from the ticket")
    suggested_action: str = Field(description="Brief suggestion for handling the ticket")


class ClassificationRequest(BaseModel):
    """Request model for ticket classification."""
    ticket_text: str = Field(..., min_length=1, max_length=10000, description="The ticket text to classify")
    channel: Optional[str] = Field(None, max_length=100, description="Channel source (e.g., email, chat)")
    
    @validator('ticket_text')
    def validate_ticket_text(cls, v):
        """Validate and sanitize ticket text."""
        if not v or not v.strip():
            raise ValueError("ticket_text cannot be empty")
        v = v.strip()
        if len(v) > 10000:
            raise ValueError("ticket_text exceeds maximum length of 10000 characters")
        return v


class BatchClassificationRequest(BaseModel):
    """Request model for batch classification."""
    messages: List[str] = Field(..., min_items=1, max_items=1000, description="List of messages to classify")
    channel: Optional[str] = Field(None, max_length=100, description="Channel source for all messages")
    
    @validator('messages')
    def validate_messages(cls, v):
        """Validate messages list."""
        if not v:
            raise ValueError("messages list cannot be empty")
        for msg in v:
            if not msg or not msg.strip():
                raise ValueError("All messages must be non-empty")
            if len(msg) > 10000:
                raise ValueError("Message exceeds maximum length of 10000 characters")
        return v

