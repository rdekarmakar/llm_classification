# main.py

"""
Main entry point for ticket classification with cost tracking.
"""

from typing import Tuple
from ticket_classifier import (
    count_tokens,
    calculate_token_cost,
    build_combined_input,
    classify_ticket_from_input,
    calculate_total_input_cost,
)
from db_manager import ChromaDBManager
from config import config
from logging_config import get_logger
from models import TicketClassification

logger = get_logger(__name__)

# Initialize DB manager singleton
db_manager = ChromaDBManager()


def classify_and_get_cost(ticket_text: str) -> Tuple[TicketClassification, float]:
    """
    Classify a ticket and calculate the total processing cost.
    
    Args:
        ticket_text: The ticket text to classify
        
    Returns:
        Tuple of (TicketClassification, total_cost)
        
    Raises:
        ValueError: If ticket_text is empty
        Exception: If classification or DB operations fail
    """
    if not ticket_text or not ticket_text.strip():
        raise ValueError("ticket_text cannot be empty")
    
    try:
        # Get collections from DB manager
        collection_customer_interaction = db_manager.get_interaction_collection()
        collection_customer_policies = db_manager.get_policies_collection()

        # Build combined input with context
        combined_input = build_combined_input(
            ticket_text,
            collection_customer_interaction,
            collection_customer_policies
        )

        # Classify ticket
        logger.info(f"Classifying ticket (length: {len(ticket_text)} chars)")
        classification = classify_ticket_from_input(combined_input)

        # Calculate input token cost
        token_stats = calculate_total_input_cost(combined_input)
        input_cost = token_stats['total_cost']

        # Calculate output token cost
        output = classification.model_dump_json(indent=2)
        output_tokens = count_tokens(output)
        output_cost = calculate_token_cost(
            output_tokens,
            config.model.output_cost_per_million
        )

        # Total cost
        total_cost = input_cost + output_cost
        
        logger.info(
            f"Classification complete: {classification.category}, "
            f"confidence: {classification.confidence:.2f}, "
            f"cost: ${total_cost:.6f}"
        )

        return classification, total_cost
        
    except Exception as e:
        logger.error(f"Error in classify_and_get_cost: {e}")
        raise
