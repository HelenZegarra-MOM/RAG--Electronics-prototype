import os
import argparse
import json
import logging
from difflib import SequenceMatcher

from classes.document_ingestor import DocumentIngestor
from classes.embedding_preparer import EmbeddingPreparer
from classes.embedding_loader import EmbeddingLoader
from classes.chromadb_retriever import ChromaDBRetriever
from classes.llm_client import LLMClient
from classes.rag_query_processor import RAGQueryProcessor
from classes.utilities import ensure_clean_dir

# Load configuration from config.json
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

# Logging setup
logging.basicConfig(
    level=getattr(logging, config.get("log_level", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main")

def step01_ingest_documents(args):
    logger.info("[Step 01] Starting document ingestion...")
    ingestor = DocumentIngestor(
        input_dir=config["raw_input_directory"],
        output_dir=config["cleaned_text_directory"]
    )
    ingestor.process_files()
    logger.info("[Step 01] Document ingestion completed.")

def step02_generate_embeddings(args):
    logger.info("[Step 02] Starting embedding generation...")
    file_list = os.listdir(config["cleaned_text_directory"])
    preparer = EmbeddingPreparer(
        file_list=file_list,
        input_dir=config["cleaned_text_directory"],
        output_dir=config["embeddings_directory"],
        embedding_model_name=config["embedding_model_name"]
    )
    preparer.process_files()
    logger.info("[Step 02] Embedding generation completed.")

def step03_store_vectors(args):
    logger.info("[Step 03] Starting vector storage...")
    ensure_clean_dir(config["vectordb_directory"])

    file_list = os.listdir(config["cleaned_text_directory"])

    loader = EmbeddingLoader(
        cleaned_text_file_list=file_list,
        cleaned_text_dir=config["cleaned_text_directory"],
        embeddings_dir=config["embeddings_directory"],
        vectordb_dir=config["vectordb_directory"],
        collection_name=config["collection_name"]
    )
    loader.process_files()
    logger.info("[Step 03] Vector storage completed.")

def step04_retrieve_relevant_chunks(args):
    logger.info("[Step 04] Starting retrieval...")
    if not args.query:
        print("❌ Please provide a query with --query")
        return

    retriever = ChromaDBRetriever(
        vectordb_dir=config["vectordb_directory"],
        embedding_model_name=config["embedding_model_name"],
        collection_name=config["collection_name"],
        score_threshold=config.get("retriever_min_score_threshold", 2.0)
    )

    results = retriever.query(args.query)

    if not results:
        print("*** No relevant documents found.")
        return

    for idx, result in enumerate(results, 1):
        print(f"\nResult {idx}:")
        print(f"ID: {result['id']}")
        print(f"Score: {result['score']}")
        print(f"Source: {result['source']}")
        print(f"Document snippet:\n{result.get('context', '')[:500]}\n{'-'*50}")
    logger.info("[Step 04] Retrieval completed.")

def step05_generate_response(args):
    logger.info("[Step 05] Generating response with RAG and direct LLM...")

    retriever = ChromaDBRetriever(
        vectordb_dir=config["vectordb_directory"],
        embedding_model_name=config["embedding_model_name"],
        collection_name=config["collection_name"],
        score_threshold=config.get("retriever_min_score_threshold", 2.0)
    )

    llm_client = LLMClient(
        llm_api_url=config["llm_api_url"],
        model_name=config["llm_model_name"]
    )

    rag_processor = RAGQueryProcessor(llm_client=llm_client, retriever=retriever, use_rag=True)
    llm_only_processor = RAGQueryProcessor(llm_client=llm_client, retriever=None, use_rag=False)

    query_text = args.query or input("Enter your query: ").strip()

    if not query_text:
        print("❌ No query provided.")
        return

    rag_response = rag_processor.query(query_text)
    llm_only_response = llm_only_processor.query(query_text)

    similarity_score = SequenceMatcher(None, rag_response, llm_only_response).ratio()

    print("\nRAG-Augmented Response:\n", rag_response)
    print("\nLLM-Only Response:\n", llm_only_response)
    print("\nSimilarity Score: {:.2f}%".format(similarity_score * 100))

    logger.info("[Step 05] Response generation comparison completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a specific step of the RAG pipeline.")
    parser.add_argument("step", choices=[
        "step01_ingest", "step02_embed", "step03_store_vectors",
        "step04_retrieve", "step05_generate_response"
    ])
    parser.add_argument("--query", help="Query string for steps 4 and 5", default=None)
    args = parser.parse_args()

    steps = {
        "step01_ingest": step01_ingest_documents,
        "step02_embed": step02_generate_embeddings,
        "step03_store_vectors": step03_store_vectors,
        "step04_retrieve": step04_retrieve_relevant_chunks,
        "step05_generate_response": step05_generate_response,
    }

    try:
        steps[args.step](args)
    except Exception as e:
        logger.error(f"Pipeline crashed during execution: {e}")
    logger.info("RAG pipeline done")
