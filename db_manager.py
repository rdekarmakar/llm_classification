"""
ChromaDB manager with singleton pattern for efficient connection management.
"""

import chromadb
from chromadb.utils import embedding_functions
from typing import Optional
import logging
from config import config

logger = logging.getLogger(__name__)


class ChromaDBManager:
    """
    Singleton manager for ChromaDB connections and collections.
    Ensures only one client instance is created and reused.
    """
    _instance: Optional['ChromaDBManager'] = None
    _client: Optional[chromadb.PersistentClient] = None
    _embedding_fn: Optional[embedding_functions.SentenceTransformerEmbeddingFunction] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._client is None:
            try:
                self._client = chromadb.PersistentClient(path=config.chroma.path)
                self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=config.chroma.embedding_model
                )
                logger.info(f"ChromaDB client initialized at path: {config.chroma.path}")
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB client: {e}")
                raise
    
    def get_collection(
        self,
        name: str,
        create_if_not_exists: bool = True,
        reset_collection: bool = False
    ) -> chromadb.Collection:
        """
        Get or create a ChromaDB collection.
        
        Args:
            name: Collection name
            create_if_not_exists: Whether to create collection if it doesn't exist
            reset_collection: Whether to delete and recreate the collection (use with caution!)
            
        Returns:
            ChromaDB collection instance
        """
        if reset_collection:
            try:
                self._client.delete_collection(name=name)
                logger.warning(f"Deleted collection: {name}")
            except Exception as e:
                logger.warning(f"Collection {name} doesn't exist or couldn't be deleted: {e}")
        
        if create_if_not_exists:
            collection = self._client.get_or_create_collection(
                name=name,
                embedding_function=self._embedding_fn
            )
            logger.debug(f"Retrieved/created collection: {name}")
            return collection
        else:
            collection = self._client.get_collection(name=name)
            logger.debug(f"Retrieved collection: {name}")
            return collection
    
    def get_interaction_collection(self, reset: bool = False) -> chromadb.Collection:
        """Get the customer interaction collection."""
        return self.get_collection(
            config.chroma.interaction_collection,
            reset_collection=reset or config.chroma.reset_on_startup
        )
    
    def get_policies_collection(self, reset: bool = False) -> chromadb.Collection:
        """Get the customer policies collection."""
        return self.get_collection(
            config.chroma.policies_collection,
            reset_collection=reset or config.chroma.reset_on_startup
        )
    
    @property
    def client(self) -> chromadb.PersistentClient:
        """Get the ChromaDB client instance."""
        return self._client
    
    def health_check(self) -> bool:
        """
        Check if ChromaDB is accessible.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            self._client.heartbeat()
            return True
        except Exception as e:
            logger.error(f"ChromaDB health check failed: {e}")
            return False

