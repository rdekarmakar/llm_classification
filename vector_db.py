# Step 1: Install ChromaDB (if not already installed)
# pip install chromadb

# Step 2: Import necessary libraries
import chromadb
from chromadb.config import Settings

# Step 3: Initialize ChromaDB client
client = chromadb.PersistentClient()

# client = chromadb.Client(Settings(
#     chroma_db_impl="duckdb+parquet",
#     persist_directory="./chroma_data"  # Directory to persist the database
# ))

# Step 4: Create or get a collection
collection = client.get_or_create_collection(name="support_tickets")

# Step 5: Example ticket classification results
ticket_classifications = [
    {
        "id": "ticket1",
        "key_information": ["Order #12345", "Received tablet instead of laptop"],
        "metadata": {
            "category": "order_issue",
            "urgency": "high",
            "sentiment": "angry"
        }
    },
    {
        "id": "ticket2",
        "key_information": ["Hospital visit on April 5th", "Insurance marked inactive", "Billed $2,300"],
        "metadata": {
            "category": "account_access",
            "urgency": "critical",
            "sentiment": "frustrated"
        }
    }
]

# Step 6: Add key_information to the ChromaDB collection
for ticket in ticket_classifications:
    for info in ticket["key_information"]:
        collection.add(
            documents=[info],
            metadatas=[ticket["metadata"]],
            ids=[f"{ticket['id']}_{info}"]
        )

# Step 7: Verify data in the collection
print(collection.get())

# results = collection.query(
#     query_texts=["Insurance inactive"], # Chroma will embed this for you
#     n_results=1 # how many results to return
# )
# print(results)
