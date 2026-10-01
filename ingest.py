import os
import pickle
import chromadb
from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(DATA_DIR, "chroma_db")
BM25_PATH = os.path.join(DATA_DIR, "bm25_index.pkl")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.pkl")

def clear_database():
    import shutil
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Database and indices cleared.")

def ingest_pdf(pdf_path, reset_existing=False):
    if reset_existing:
        clear_database()
    print(f"Ingesting {pdf_path}...")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 1. Extraction using Docling
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    document_text = result.document.export_to_markdown()
    
    # 2. Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_text(document_text)
    print(f"Extracted {len(chunks)} chunks.")
    
    # 3. Vector DB Setup (ChromaDB)
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    
    from chromadb.utils import embedding_functions
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    collection = chroma_client.get_or_create_collection(
        name="support_docs", 
        embedding_function=sentence_transformer_ef
    )
    
    start_idx = collection.count()
    ids = [f"chunk_{start_idx + i}" for i in range(len(chunks))]
    metadatas = [{"source": pdf_path, "chunk_index": i} for i in range(len(chunks))]
    
    collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=ids
    )
    
    # 4. BM25 Index Setup
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, 'rb') as f:
            all_chunks = pickle.load(f)
    else:
        all_chunks = []
        
    all_chunks.extend(chunks)
    
    tokenized_corpus = [chunk.lower().split(" ") for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    
    with open(BM25_PATH, 'wb') as f:
        pickle.dump(bm25, f)
        
    with open(CHUNKS_PATH, 'wb') as f:
        pickle.dump(all_chunks, f)
        
    print("Ingestion complete. ChromaDB and BM25 index saved.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        ingest_pdf(sys.argv[1])
