import logging
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logger = logging.getLogger(__name__)

class ChromaDBRetriever:
    def __init__(self, vectordb_dir, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", collection_name="collections", score_threshold=3.5):
        self.vectordb_dir = vectordb_dir
        self.collection_name = collection_name
        self.score_threshold = score_threshold

        self.client = PersistentClient(path=vectordb_dir)
        self.embedding_function = SentenceTransformerEmbeddingFunction(model_name=embedding_model_name)
        self.collection = self.client.get_or_create_collection(name=collection_name, embedding_function=self.embedding_function)

    def query(self, query_text, top_k=3):
        logger.info(f"🔍 Querying ChromaDB with: {query_text}")

        results = self.collection.query(query_texts=[query_text], n_results=top_k)
        matches = []

        if "distances" not in results or not results["distances"]:
            return []

        for i, distance in enumerate(results["distances"][0]):
            if distance <= self.score_threshold:
                match = {
                    "id": results["ids"][0][i],
                    "score": distance,
                    "context": results["documents"][0][i] if results["documents"][0] else None,
                    "source": results["metadatas"][0][i].get("source", "Unknown")
                }
                logger.debug(f"Accepted match: {match['id']} with score {match['score']:.4f}")
                matches.append(match)
            else:
                logger.debug(f"Skipping '{results['ids'][0][i]}' — distance {distance:.2f} > threshold {self.score_threshold}")

        logger.info(f"Retrieved {len(matches)} relevant document(s) after filtering.")
        return matches
