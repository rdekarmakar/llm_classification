# ticket_classifier.py

"""
This code defines a structured data model for classifying customer support tickets using Pydantic and Python's Enum class. 
It specifies categories, urgency levels, customer sentiments, and other relevant information as predefined options or constrained fields. 
This structure ensures data consistency, enables automatic validation, and facilitates easy integration with AI models and other parts of a support ticket system.
"""

from typing import List
import tiktoken
from pydantic import BaseModel, Field
from enum import Enum
import instructor
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# -------------------------------
# Enums and Pydantic models
# -------------------------------
class TicketCategory(str, Enum):
    ORDER_ISSUE = "claim_denial"
    ACCOUNT_ACCESS = "account_access"
    PRODUCT_INQUIRY = "coverage_inquiry"
    TECHNICAL_SUPPORT = "dependent_coverage_issue"
    BILLING = "billing_issue"
    OTHER = "other"

class CustomerSentiment(str, Enum):
    ANGRY = "angry"
    FRUSTRATED = "frustrated"
    NEUTRAL = "neutral"
    SATISFIED = "satisfied"

class TicketUrgency(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TicketClassification(BaseModel):
    category: TicketCategory
    urgency: TicketUrgency
    sentiment: CustomerSentiment
    confidence: float = Field(ge=0, le=1, description="Confidence score for the classification")
    key_information: List[str] = Field(description="List of key points extracted from the ticket")
    suggested_action: str = Field(description="Brief suggestion for handling the ticket")

# -------------------------------
# Utilities
# -------------------------------
def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def calculate_token_cost(token_count: int, cost_per_million_tokens: float) -> float:
    return (token_count * cost_per_million_tokens) / 1_000_000

def build_combined_input(ticket_text: str, interaction_collection='', policy_collection='') -> str:
    results = interaction_collection.query(query_texts=[ticket_text], n_results=1)
    interaction_context = " ".join([doc for sublist in results["documents"] for doc in sublist])

    results = policy_collection.query(query_texts=[ticket_text], n_results=1)
    policy_context = " ".join([doc for sublist in results["documents"] for doc in sublist])

    additional_context = f"{interaction_context} {policy_context}".strip()
    return f"{ticket_text}\n\nAdditional Context:\n{additional_context}"

# -------------------------------
# Classification function
# -------------------------------

SYSTEM_PROMPT = """
You are an AI assistant for a large health insurance customer support team. 
Your role is to analyze incoming customer support requests and provide structured information to help our team respond quickly and effectively.
Business Context:
- We handle thousands of requests daily across various categories (claim, accounts, products, technical issues, billing).
- Quick and accurate classification is crucial for customer satisfaction and operational efficiency.
- We prioritize based on urgency and customer sentiment.
Your tasks:
1. Categorize the requests into the most appropriate category.
2. Assess the urgency of the issue (low, medium, high, critical).
3. Determine the customer's sentiment.
4. Extract key information that would be helpful for our support team.
5. Suggest an initial action for handling the ticket.
6. Provide a confidence score for your classification.
Remember:
- Be objective and base your analysis solely on the information provided in the ticket.
- If you're unsure about any aspect, reflect that in your confidence score.
- For 'key_information', extract specific details like Policy numbers, product names,current issues or brief from previous customer interactions.
- The 'suggested_action' should be a brief, actionable step for our support team.
Analyze the following customer support requests and provide the requested information in the specified format.
As additional context, you can use the customer interaction history and customer policies.
"""

# Patch instructor to the Groq client
groq_client = instructor.from_groq(Groq())

def classify_ticket_from_input(combined_input: str) -> TicketClassification:
    response = groq_client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        response_model=TicketClassification,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": combined_input}
        ]
    )
    return response

def get_system_prompt() -> str:
    return SYSTEM_PROMPT

def calculate_total_input_cost(combined_input: str, model: str = "gpt-3.5-turbo", cost_per_million_tokens: float = 0.15) -> dict:
    system_prompt = get_system_prompt()
    system_prompt_tokens = count_tokens(system_prompt, model)
    input_tokens = count_tokens(combined_input, model)
    total_tokens = input_tokens + system_prompt_tokens
    total_cost = calculate_token_cost(total_tokens, cost_per_million_tokens)
    return {
        "system_prompt_tokens": system_prompt_tokens,
        "input_tokens": input_tokens,
        "total_tokens": total_tokens,
        "total_cost": total_cost
    }
