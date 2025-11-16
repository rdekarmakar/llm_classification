# ticket_classifier.py

"""
Customer Support Ticket Classification System with LLM integration.
Uses structured output via Instructor and Pydantic for reliable classification.
"""

from typing import List, Optional
import tiktoken
import instructor
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

from models import TicketClassification, TicketCategory, CustomerSentiment, TicketUrgency
from config import config
from logging_config import get_logger

logger = get_logger(__name__)

# -------------------------------
# Utilities
# -------------------------------
def count_tokens(text: str, model: Optional[str] = None) -> int:
    """
    Count tokens in text using tiktoken.
    
    Args:
        text: Text to count tokens for
        model: Model name for tokenizer (defaults to config)
        
    Returns:
        Number of tokens
    """
    if model is None:
        model = config.model.token_count_model
    
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Failed to count tokens with model {model}, using cl100k_base: {e}")
        # Fallback encoding
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))


def calculate_token_cost(token_count: int, cost_per_million_tokens: float) -> float:
    """
    Calculate cost for token usage.
    
    Args:
        token_count: Number of tokens
        cost_per_million_tokens: Cost per million tokens
        
    Returns:
        Total cost
    """
    return (token_count * cost_per_million_tokens) / 1_000_000


def build_combined_input(
    ticket_text: str,
    interaction_collection,
    policy_collection,
    n_results: Optional[int] = None
) -> str:
    """
    Build combined input with context from vector database.
    
    Args:
        ticket_text: Original ticket text
        interaction_collection: ChromaDB collection for interactions
        policy_collection: ChromaDB collection for policies
        n_results: Number of results to retrieve (defaults to config)
        
    Returns:
        Combined input string with context
    """
    if not ticket_text or not ticket_text.strip():
        raise ValueError("ticket_text cannot be empty")
    
    if n_results is None:
        n_results = config.chroma.query_n_results
    
    try:
        # Query interaction history
        interaction_results = interaction_collection.query(
            query_texts=[ticket_text],
            n_results=n_results
        )
        interaction_context = " ".join([
            doc for sublist in interaction_results.get("documents", [])
            for doc in sublist
        ])
        
        # Query policy information
        policy_results = policy_collection.query(
            query_texts=[ticket_text],
            n_results=n_results
        )
        policy_context = " ".join([
            doc for sublist in policy_results.get("documents", [])
            for doc in sublist
        ])
        
        additional_context = f"{interaction_context} {policy_context}".strip()
        
        if additional_context:
            return f"{ticket_text}\n\nAdditional Context:\n{additional_context}"
        else:
            logger.debug("No additional context found, using ticket text only")
            return ticket_text
            
    except Exception as e:
        logger.error(f"Error building combined input: {e}")
        # Fallback to just ticket text if context retrieval fails
        return ticket_text

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
- For 'key_information', extract specific details like Policy numbers, product names, current issues or brief from previous customer interactions.
- The 'suggested_action' should be a brief, actionable step for our support team.
Analyze the following customer support requests and provide the requested information in the specified format.
As additional context, you can use the customer interaction history and customer policies.
"""

# Initialize Groq client with instructor patch
try:
    groq_client = instructor.from_groq(Groq(api_key=config.groq_api_key))
    logger.info("Groq client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {e}")
    raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def classify_ticket_from_input(combined_input: str) -> TicketClassification:
    """
    Classify a ticket from combined input (ticket + context).
    
    Args:
        combined_input: Combined ticket text with additional context
        
    Returns:
        TicketClassification object with structured results
        
    Raises:
        Exception: If classification fails after retries
    """
    if not combined_input or not combined_input.strip():
        raise ValueError("combined_input cannot be empty")
    
    try:
        logger.debug(f"Classifying ticket (length: {len(combined_input)} chars)")
        response = groq_client.chat.completions.create(
            model=config.model.name,
            response_model=TicketClassification,
            temperature=config.model.temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": combined_input}
            ]
        )
        logger.debug(f"Classification successful: {response.category}, confidence: {response.confidence}")
        return response
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise

def get_system_prompt() -> str:
    """Get the system prompt used for classification."""
    return SYSTEM_PROMPT


def calculate_total_input_cost(
    combined_input: str,
    model: Optional[str] = None,
    cost_per_million_tokens: Optional[float] = None
) -> dict:
    """
    Calculate total input cost including system prompt.
    
    Args:
        combined_input: Combined input text
        model: Model name for token counting (defaults to config)
        cost_per_million_tokens: Cost per million tokens (defaults to config)
        
    Returns:
        Dictionary with token counts and costs
    """
    if model is None:
        model = config.model.token_count_model
    if cost_per_million_tokens is None:
        cost_per_million_tokens = config.model.input_cost_per_million
    
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
