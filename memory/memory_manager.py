import uuid
from typing import List

import decouple
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

open_ai_embeddings = OpenAIEmbeddings(openai_api_key=decouple.config("API_KEY"))
recall_vector_store = InMemoryVectorStore(open_ai_embeddings)

def get_user_id(config: RunnableConfig) -> str:
    user_id = config["configurable"].get("user_id")
    if user_id is None:
        raise ValueError("User ID needs to be provided to save a memory.")

    return user_id

class GraphMemory:
    @staticmethod
    @tool
    def save_recall_memory(memory: str, config: RunnableConfig) -> str:
        """Save memory to vectorstore for later semantic retrieval."""
        user_id = get_user_id(config)
        document = Document(
            page_content=memory, id=str(uuid.uuid4()), metadata={"user_id": user_id}
        )
        recall_vector_store.add_documents([document])
        return memory

    @staticmethod
    @tool
    def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
        """Search for relevant memories."""
        user_id = get_user_id(config)

        def _filter_function(doc: Document) -> bool:
            return doc.metadata.get("user_id") == user_id

        documents = recall_vector_store.similarity_search(
            query, k=3, filter=_filter_function
        )
        return [document.page_content for document in documents]