import asyncio
import structlog
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore  # Updated Import
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Assuming your Config class exists in config.py
from config import Config

settings = Config()
logger = structlog.get_logger()

async def safe_openai_embeddings():
    try:
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.openai_api_key
        )
    except Exception as e:
        logger.error("Failed to create OpenAI embeddings", error=str(e))
        raise

async def pdf_loader(docs_path="./docs"):
    docs_path_obj = Path(docs_path)
    if not docs_path_obj.exists():
        logger.error(f"❌ Directory {docs_path} does not exist")
        return None
        
    pdf_files = list(docs_path_obj.glob("**/*.pdf"))
    if not pdf_files:
        logger.warning("⚠️ No PDF files found in docs/!")
        return None

    all_docs = []
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_file"] = pdf_file.name
            all_docs.extend(docs)
            logger.info(f"✅ Loaded {len(docs)} pages from {pdf_file.name}")
        except Exception as e:
            logger.error(f"❌ Failed loading {pdf_file.name}: {str(e)}")
    
    return all_docs if all_docs else None

async def safe_text_splitter(all_docs):
    if not all_docs: return None
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = splitter.split_documents(all_docs)
        logger.info(f"✂️ Created {len(splits)} chunks")
        return splits
    except Exception as e:
        logger.error("❌ Text splitting failed", error=str(e))
        return None

async def safe_indexer(splits, embeddings):
    """
    Refactored to use the modern QdrantVectorStore.from_documents
    This replaces manual upserting and point management.
    """
    try:
        logger.info(f"📈 Indexing {len(splits)} chunks into Qdrant...")

        # from_documents handles collection creation, embedding, and upserting automatically
        vectorstore = QdrantVectorStore.from_documents(
            documents=splits,
            embedding=embeddings,
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection_name=settings.collection_name,
            # If you want to overwrite every time, set force_recreate=True
            force_recreate=False 
        )

        logger.info(f"✅ SUCCESS: {len(splits)} chunks indexed!")
        return vectorstore
        
    except Exception as e:
        logger.error(f"❌ Indexing failed: {str(e)}")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def create_qdrant_vector_store():
    try:
        # 1. Get Embeddings
        embeddings = await safe_openai_embeddings()
        
        # 2. Load Docs
        docs = await pdf_loader()
        if not docs:
            logger.error("No documents to process")
            return None

        # 3. Split Docs
        splits = await safe_text_splitter(docs)
        if not splits:
            logger.error("No splits created")
            return None

        # 4. Index and return VectorStore
        vectorstore = await safe_indexer(splits, embeddings)
        return vectorstore

    except Exception as e:
        logger.error("Main processing pipeline failed", error=str(e))
        raise

# if __name__ == "__main__":
#     vector_db = asyncio.run(create_qdrant_vector_store())
#     if vector_db:
#         print("Pipeline completed successfully.")