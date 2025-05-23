
import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(path="my_vectordb")

collection = client.get_or_create_collection(name="my_collection")

# client.delete_collection(name="my_collection")

results = collection.query(
    query_texts=["annual health check john doe"], # Chroma will embed this for you
    n_results=2 # how many results to return
)
print(results['documents'])