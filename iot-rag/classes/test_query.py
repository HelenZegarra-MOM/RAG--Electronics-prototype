import argparse
from chromadb_retriever import ChromaDBRetriever

def run_test_query(query_text):
    retriever = ChromaDBRetriever(
        vectordb_dir="data/vectordb",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        collection_name="collections",
        score_threshold=2.0  # Adjust as needed
    )

    results = retriever.query(query_text, top_k=3)

    print(f"\n🔍 Query: {query_text}")
    print(f"📄 Retrieved {len(results)} result(s):\n")

    if not results:
        print("⚠️ No relevant documents found.")
    else:
        for idx, result in enumerate(results):
            doc_id = result.get("id", "N/A")
            score = result.get("score", 0.0)
            source = result.get("source", "N/A")
            context = result.get("context", "").replace("\n", " ").strip()

            print(f"--- Result {idx + 1} ---")
            print(f"ID: {doc_id}")
            print(f"Score: {score:.3f}")
            print(f"Source: {source}")
            print(f"Context Preview: {context[:250]}...\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a test query against ChromaDB.")
    parser.add_argument(
        "--query",
        type=str,
        default="What is the recommended operating temperature for the ATMEGA2560?",
        help="Query string to search the vector database.",
    )
    args = parser.parse_args()
    run_test_query(args.query)
