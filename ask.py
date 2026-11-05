from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate

COLLECTION_NAME = "enterprise_knowledge"
QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"

_vector_store = None
_llm = None
_prompt = None


def _get_vector_store():
    global _vector_store

    if _vector_store is None:
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=OLLAMA_URL,
        )
        try:
            _vector_store = QdrantVectorStore.from_existing_collection(
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                url=QDRANT_URL,
            )
        except Exception as e:
            raise RuntimeError(
                "Knowledge base not found. "
                "Run `python ingest.py` after starting Qdrant "
                "(docker compose up -d) and Ollama."
            ) from e

    return _vector_store


def _get_llm():
    global _llm

    if _llm is None:
        _llm = ChatOllama(
            model="qwen2.5:7b",
            temperature=0,
            base_url=OLLAMA_URL,
        )

    return _llm


def _get_prompt():
    global _prompt

    if _prompt is None:
        _prompt = ChatPromptTemplate.from_template("""
You are an enterprise knowledge assistant.

Answer the question using ONLY the provided context.

If the answer is not available in the context,
say that you could not find sufficient information.

Do not invent facts.

Context:
{context}

Question:
{question}

Provide a clear and structured answer.
Mention the source document and page where available.
""")

    return _prompt


def ask_question(question):
    vector_store = _get_vector_store()
    llm = _get_llm()
    prompt = _get_prompt()

    results = vector_store.similarity_search_with_score(
        question,
        k=4,
    )

    context = "\n\n".join(
        [
            f"Source: {doc.metadata.get('source')}, "
            f"Page: {doc.metadata.get('page', 'N/A')}\n"
            f"{doc.page_content}"
            for doc, score in results
        ]
    )

    messages = prompt.format_messages(
        context=context,
        question=question,
    )

    response = llm.invoke(messages)

    print("\nAI ANSWER:\n")
    print(response.content)

    print("\nRETRIEVED DOCUMENTS:\n")

    for doc, score in results:
        print(
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            "Score:",
            score,
        )

    return response.content


if __name__ == "__main__":
    while True:
        question = input("\nAsk a question (exit to quit): ")

        if question.lower() == "exit":
            break

        ask_question(question)
