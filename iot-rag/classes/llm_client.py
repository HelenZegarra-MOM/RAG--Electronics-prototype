# classes/llm_client.py

import requests
import logging
import json

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, llm_api_url: str, model_name: str):
        self.llm_api_url = llm_api_url
        self.model_name = model_name

    def query(self, prompt: str, context: str = None) -> str:
        """
        Sends a prompt to the LLM endpoint and returns the response.
        If context is provided, it uses RAG-style prompting.
        """
        try:
            messages = []

            # Truncate context if it's too long (optional safeguard)
            if context and len(context) > 4000:
                logger.warning("⚠️ Context too long, truncating...")
                context = context[:4000]

            if context:
                messages.append({
                    "role": "system",
                    "content": "You are a helpful assistant. Use the context to answer the user's question. If context is insufficient, say so clearly."
                })
                messages.append({
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {prompt}"
                })
            else:
                messages.append({
                    "role": "user",
                    "content": prompt
                })

            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.5,
                "stream": False
            }

            logger.debug("🔍 Sending payload to LLM:\n%s", json.dumps(payload, indent=2))

            response = requests.post(
                url=f"{self.llm_api_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=90
            )
            response.raise_for_status()

            result = response.json()
            logger.debug("✅ LLM raw response: %s", result)

            return result["choices"][0]["message"]["content"].strip()

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error connecting to LLM endpoint: {e}")
            return "Error: Could not connect to the LLM endpoint."
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"❌ Error parsing LLM response: {e}")
            return "Error: Invalid response from LLM."
