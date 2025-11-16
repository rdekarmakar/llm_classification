"""
Batch classification with vector database storage.
"""

from main import classify_and_get_cost
from message_router import MessageRouter
from text_normalize import normalize_text2
from db_manager import ChromaDBManager
from config import config
from logging_config import get_logger

import pandas as pd
import uuid
from typing import List, Tuple

logger = get_logger(__name__)

# Initialize DB manager singleton
db_manager = ChromaDBManager()

# Get the collection (only reset if explicitly configured)
collection = db_manager.get_interaction_collection(reset=config.chroma.reset_on_startup)

def classify(logs: List[Tuple[str, str]]) -> Tuple[List[str], List[str], List[float], List[str]]:
    """
    Classify multiple log entries and store them in vector database.
    
    Args:
        logs: List of tuples (channel, message_content)
        
    Returns:
        Tuple of (labels, routing_info, processing_costs, log_ids)
    """
    if not logs:
        logger.warning("Empty logs list provided")
        return [], [], [], []
    
    labels = []
    processing_costs = []
    routing_info = []
    log_ids = []

    documents = []
    metadatas = []
    ids = []

    logger.info(f"Processing {len(logs)} log entries")
    
    for i, (channel, message_content) in enumerate(logs):
        try:
            # Normalize and prepare for Chroma
            norm_msg = normalize_text2(message_content)
            doc_id = str(uuid.uuid4())
            documents.append(norm_msg)
            metadatas.append({"channel": channel or "unknown"})
            ids.append(doc_id)
            log_ids.append(doc_id)

            # Classify and compute cost
            label, cost = classify_log(channel, message_content)
            labels.append(label)
            processing_costs.append(cost)

            # Routing
            router = MessageRouter(label)
            routing_info.append(router.display_routing())
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(logs)} entries")
                
        except Exception as e:
            logger.error(f"Error processing log entry {i}: {e}")
            # Add placeholder values to maintain list alignment
            labels.append("{}")
            processing_costs.append(0.0)
            routing_info.append("Error processing")
            log_ids.append(str(uuid.uuid4()))

    # Add to ChromaDB in batch
    try:
        if documents:
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
            logger.info(f"Added {len(documents)} documents to ChromaDB")
    except Exception as e:
        logger.error(f"Error adding documents to ChromaDB: {e}")
        # Continue even if DB storage fails

    return labels, routing_info, processing_costs, log_ids

def classify_log(channel: str, message_content: str) -> Tuple[str, float]:
    """
    Classify a single log entry.
    
    Args:
        channel: Channel source
        message_content: Message content to classify
        
    Returns:
        Tuple of (JSON label string, total_cost)
    """
    try:
        classification, total_cost = classify_and_get_cost(message_content)
        label = classification.model_dump_json(indent=2)
        return label, total_cost
    except Exception as e:
        logger.error(f"Error classifying log (channel: {channel}): {e}")
        raise


def classify_csv(input_file: str, output_file: str = "output_with_chroma.csv") -> str:
    """
    Classify tickets from a CSV file.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        
    Returns:
        Path to output file
    """
    try:
        df = pd.read_csv(input_file, encoding='ISO-8859-1')
        
        # Validate required columns
        required_columns = ["channel", "message_content"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"CSV missing required columns: {missing_columns}")

        # Classify based on 'channel' and 'message_content'
        logs = list(zip(df["channel"], df["message_content"]))
        labels, routing_info, processing_costs, chroma_ids = classify(logs)

        # Append results
        df["target_label"] = labels
        df["routing_info"] = routing_info
        df["processing_cost"] = processing_costs
        df["chroma_vector_id"] = chroma_ids

        logger.info(f"Classification complete. Results preview:\n{df.head()}")
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")

        return output_file
        
    except Exception as e:
        logger.error(f"Error in classify_csv: {e}")
        raise

if __name__ == '__main__':
    classify_csv("test.csv")
