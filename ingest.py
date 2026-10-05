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
CHUNKS_PATH = os.path.join(DATA_DIR, "child_chunks.pkl")
PARENTS_PATH = os.path.join(DATA_DIR, "parents.pkl")

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
    
    # 2. Parent-Child Chunking
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    
    parent_chunks = parent_splitter.split_text(document_text)
    
    if os.path.exists(PARENTS_PATH):
        with open(PARENTS_PATH, 'rb') as f:
            all_parents = pickle.load(f)
    else:
        all_parents = {}
        
    start_parent_id = len(all_parents)
    
    new_child_chunks = []
    new_child_metadatas = []
    
    for i, p_chunk in enumerate(parent_chunks):
        p_id = f"parent_{start_parent_id + i}"
        all_parents[p_id] = p_chunk
        
        c_chunks = child_splitter.split_text(p_chunk)
        for c_chunk in c_chunks:
            new_child_chunks.append(c_chunk)
            new_child_metadatas.append({"source": pdf_path, "parent_id": p_id})
            
    print(f"Extracted {len(parent_chunks)} parent chunks and {len(new_child_chunks)} child chunks.")
    
    # 3. Vector DB Setup (ChromaDB)
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    
    from chromadb.utils import embedding_functions
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    collection = chroma_client.get_or_create_collection(
        name="support_docs", 
        embedding_function=sentence_transformer_ef
    )
    
    start_idx = collection.count()
    ids = [f"child_{start_idx + i}" for i in range(len(new_child_chunks))]
    
    collection.add(
        documents=new_child_chunks,
        metadatas=new_child_metadatas,
        ids=ids
    )
    
    # 4. BM25 Index Setup
    if os.path.exists(CHUNKS_PATH):
        with open(CHUNKS_PATH, 'rb') as f:
            all_child_data = pickle.load(f)
    else:
        all_child_data = []
        
    for text, meta in zip(new_child_chunks, new_child_metadatas):
        all_child_data.append({"text": text, "parent_id": meta["parent_id"]})
    
    tokenized_corpus = [item["text"].lower().split(" ") for item in all_child_data]
    bm25 = BM25Okapi(tokenized_corpus)
    
    with open(BM25_PATH, 'wb') as f:
        pickle.dump(bm25, f)
        
    with open(CHUNKS_PATH, 'wb') as f:
        pickle.dump(all_child_data, f)
        
    with open(PARENTS_PATH, 'wb') as f:
        pickle.dump(all_parents, f)
        
    print("Ingestion complete. ChromaDB, BM25, and Parent definitions saved.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        ingest_pdf(sys.argv[1])
