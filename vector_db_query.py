import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(path="my_vectordb")

collection = client.get_or_create_collection(name="customer_policies")

# client.delete_collection(name="my_collection")
ticket_text = """
Charlie Davis :I'm planning to have a minor outpatient surgery next month and need to confirm if it's covered under my current plan.
Can you please send me details of what's included in my benefits and any pre-authorization requirements?
"""

results = collection.query(
    query_texts=[ticket_text], # Chroma will embed this for you
    n_results=1 # how many results to return
)
print(results['documents'])