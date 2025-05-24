# from intent_prediction2 import classify_ticket
from main import classify_and_get_cost
from message_router import MessageRouter
from text_normalize import normalize_text2

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import uuid

# Setup ChromaDB persistent client
chroma_client = chromadb.PersistentClient(path="my_vectordb")

# Define the embedding function
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")

# Uncomment to reset collection
chroma_client.delete_collection(name="customer_interaction")

# Get or create the collection
collection = chroma_client.get_or_create_collection(name="customer_interaction", embedding_function=embedding_fn)

def classify(logs):
    labels = []
    processing_costs = []
    routing_info = []
    log_ids = []

    documents = []
    metadatas = []
    ids = []

    for i, (channel, message_content) in enumerate(logs):
        # Normalize and prepare for Chroma
        norm_msg = normalize_text2(message_content)
        doc_id = str(uuid.uuid4())
        documents.append(norm_msg)
        metadatas.append({"channel": channel})
        ids.append(doc_id)
        log_ids.append(doc_id)

        # Classify and compute cost
        label, cost = classify_log(channel, message_content)
        labels.append(label)
        processing_costs.append(cost)

        # Routing
        router = MessageRouter(label)
        routing_info.append(router.display_routing())

    # Add to ChromaDB
    collection.add(documents=documents, metadatas=metadatas, ids=ids)

    return labels, routing_info, processing_costs, log_ids

def classify_log(channel, message_content):
    classification, total_cost = classify_and_get_cost(message_content)
    label = classification.model_dump_json(indent=2)
    return label, total_cost

def classify_csv(input_file):
    df = pd.read_csv(input_file, encoding='ISO-8859-1')

    # Classify based on 'channel' and 'message_content'
    logs = list(zip(df["channel"], df["message_content"]))
    labels, routing_info, processing_costs, chroma_ids = classify(logs)

    # Append results
    df["target_label"] = labels
    df["routing_info"] = routing_info
    df["processing_cost"] = processing_costs
    df["chroma_vector_id"] = chroma_ids

    print(df.head())

    output_file = "output_with_chroma.csv"
    df.to_csv(output_file, index=False)

    return output_file

if __name__ == '__main__':
    classify_csv("test.csv")
