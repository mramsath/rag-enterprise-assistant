
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

DOCUMENTS_DIR = Path("documents")

COLLECTION_NAME = "enterprise_knowledge"

# 1. Initialize embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# 2. Split large documents into smaller chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

all_chunks = []

# 3. Load PDF documents
for pdf_file in DOCUMENTS_DIR.glob("*.pdf"):

    print(f"Processing: {pdf_file.name}")

    loader = PyPDFLoader(str(pdf_file))

    pages = loader.load()

    # 4. Split extracted content
    chunks = splitter.split_documents(pages)

    # 5. Add metadata
    for chunk in chunks:
        chunk.metadata["source"] = pdf_file.name

    all_chunks.extend(chunks)

print(f"Total chunks: {len(all_chunks)}")

if not all_chunks:
    raise ValueError("No PDF content found.")

# 6. Convert chunks into embeddings
# 7. Store vectors and document content in Qdrant

vector_store = QdrantVectorStore.from_documents(
    documents=all_chunks,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    url="http://localhost:6333",
    force_recreate=False
)

print("Documents successfully indexed!")
