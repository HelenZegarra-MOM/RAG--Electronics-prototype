import logging
from .llm_client import LLMClient
from .chromadb_retriever import ChromaDBRetriever

class RAGQueryProcessor:
    def __init__(
        self,
        llm_client: LLMClient,
        retriever: ChromaDBRetriever,
        use_rag: bool = False
    ):
        self.use_rag = use_rag
        self.llm_client = llm_client
        self.retriever = retriever if use_rag else None
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized RAGQueryProcessor | use_rag={use_rag}")

    def query(self, query_text: str) -> str:
        self.logger.info(f"Processing query: {query_text}")
        context = ""
        metadata_info = {}

        if self.use_rag and self.retriever:
            self.logger.info("RAG enabled: retrieving relevant documents...")
            retrieved_docs = self.retriever.query(query_text)

            if not retrieved_docs:
                self.logger.warning("No relevant documents retrieved.")
            else:
                result = retrieved_docs[0]
                context = result.get("context", "")
                metadata_info = {
                    "id": result.get("id"),
                    "score": result.get("score"),
                    "source": result.get("source")
                }

                preview = (context[:150] + "...") if len(context) > 150 else context
                self.logger.info(f"Top match | ID: {metadata_info['id']} | Score: {metadata_info['score']} | Source: {metadata_info['source']}")
                self.logger.debug(f"Context Preview: {preview}")

        response = self.llm_client.query(prompt=query_text, context=context)
        self.logger.debug(f"LLM Response: {response}")
        return response

    def generate_response(self, query_text: str) -> dict:
        self.logger.info("🧠 Running full RAG comparison...")
        rag_response = self.query(query_text)
        llm_only_response = self.llm_client.query(prompt=query_text)

        similarity_score = 0.0
        if self.use_rag and self.retriever:
            retrieved_docs = self.retriever.query(query_text)
            if retrieved_docs:
                similarity_score = max(doc.get("score", 0.0) for doc in retrieved_docs)

        return {
            "rag_response": rag_response,
            "llm_response": llm_only_response,
            "similarity_score": round(similarity_score, 2)
        }
