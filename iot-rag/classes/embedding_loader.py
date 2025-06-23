import os
import json
import logging
from typing import List
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logger = logging.getLogger(__name__)

class EmbeddingLoader:
    def __init__(
        self,
        cleaned_text_file_list: List[str],
        cleaned_text_dir: str,
        embeddings_dir: str,
        vectordb_dir: str,
        collection_name: str
    ):
        self.cleaned_text_file_list = cleaned_text_file_list
        self.cleaned_text_dir = cleaned_text_dir
        self.embeddings_dir = embeddings_dir
        self.vectordb_dir = vectordb_dir
        self.collection_name = collection_name

    def process_files(self):
        logger.info("🔃 Starting vector store creation...")

        client = PersistentClient(path=self.vectordb_dir)
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-MiniLM-L6-v2")
        collection = client.get_or_create_collection(name=self.collection_name, embedding_function=embedding_fn)

        for filename in self.cleaned_text_file_list:
            base_name = os.path.splitext(filename)[0]
            embedding_file = os.path.join(self.embeddings_dir, f"{base_name}_embeddings.json")

            if not os.path.exists(embedding_file):
                logger.warning(f"❌ Embedding file not found: {embedding_file}")
                continue

            with open(embedding_file, "r", encoding="utf-8") as f:
                try:
                    embedding_data = json.load(f)
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Error parsing {embedding_file}: {e}")
                    continue

            # Handle both old list format and proper dict format
            if isinstance(embedding_data, list):
                embeddings = embedding_data
                metadata = {}
            else:
                embeddings = embedding_data.get("embeddings", [])
                metadata = embedding_data.get("metadata", {})

            text_file_path = os.path.join(self.cleaned_text_dir, filename)
            if not os.path.exists(text_file_path):
                logger.warning(f"⚠️ Text file not found: {text_file_path}")
                continue

            with open(text_file_path, "r", encoding="utf-8") as tf:
                content = tf.read()

            doc_id = f"doc_{base_name}"
            collection.add(
                documents=[content],
                embeddings=embeddings,
                metadatas=[{"source": filename, **metadata}],
                ids=[doc_id]
            )

            logger.info(f"✅ Added {filename} to vector DB as {doc_id}")

        logger.info("📚 Vector storage completed.")
