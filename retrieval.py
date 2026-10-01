import os
import pickle
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder
from anthropic import Anthropic
import openai
import google.generativeai as genai

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(DATA_DIR, "chroma_db")
BM25_PATH = os.path.join(DATA_DIR, "bm25_index.pkl")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.pkl")

def reciprocal_rank_fusion(semantic_results, bm25_results, k=60):
    rrf_scores = {}
    
    for rank, chunk in enumerate(semantic_results):
        if chunk not in rrf_scores:
            rrf_scores[chunk] = 0
        rrf_scores[chunk] += 1 / (k + rank + 1)
        
    for rank, chunk in enumerate(bm25_results):
        if chunk not in rrf_scores:
            rrf_scores[chunk] = 0
        rrf_scores[chunk] += 1 / (k + rank + 1)
        
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in sorted_results]

def get_hybrid_results(query, top_k=20):
    if not os.path.exists(DB_DIR):
        return []

    # Load ChromaDB
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = chroma_client.get_collection(name="support_docs", embedding_function=sentence_transformer_ef)
    
    # Semantic Search
    semantic_res = collection.query(query_texts=[query], n_results=top_k)
    semantic_chunks = semantic_res['documents'][0] if semantic_res['documents'] else []
    
    # Load BM25 & Chunks
    if not os.path.exists(BM25_PATH) or not os.path.exists(CHUNKS_PATH):
        return semantic_chunks

    with open(BM25_PATH, 'rb') as f:
        bm25 = pickle.load(f)
    with open(CHUNKS_PATH, 'rb') as f:
        all_chunks = pickle.load(f)
        
    # Keyword Search
    tokenized_query = query.lower().split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    
    import numpy as np
    top_n_indices = np.argsort(bm25_scores)[::-1][:top_k]
    bm25_chunks = [all_chunks[i] for i in top_n_indices]
    
    # RRF
    rrf_chunks = reciprocal_rank_fusion(semantic_chunks, bm25_chunks)
    return rrf_chunks[:top_k]

_CROSS_ENCODER = None

def get_cross_encoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        _CROSS_ENCODER = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
    return _CROSS_ENCODER

def rerank_results(query, chunks, top_n=5):
    if not chunks:
        return []
    
    # Use cached Cross-Encoder
    model = get_cross_encoder()
    
    # Score chunks
    pairs = [[query, chunk] for chunk in chunks]
    scores = model.predict(pairs)
    
    # Sort chunks by score
    scored_chunks = list(zip(chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    return [chunk for chunk, score in scored_chunks[:top_n]]

def generate_answer(query, context_chunks, model_choice="claude"):
    if not context_chunks:
        return "I cannot answer this based on the provided documents as no relevant context was found."

    context_str = "\n\n---\n\n".join(context_chunks)
    
    system_prompt = (
        "You are a helpful Support Desk AI. Answer the user's question ONLY using the provided context. "
        "If the answer is not in the context, say 'I cannot answer this based on the provided documents.'\n\n"
        f"Context:\n{context_str}"
    )
    
    if model_choice.lower() == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            return "Error: Gemini API Key is missing. Please enter your API Key in the sidebar."
        
        genai.configure(api_key=api_key)
        
        candidate_models = []
        try:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    candidate_models.append(m.name)
        except Exception:
            pass

        # Fallback names if list_models is restricted
        backups = [
            'gemini-1.5-flash-latest', 
            'gemini-2.0-flash', 
            'gemini-2.5-flash',
            'gemini-1.5-flash-001', 
            'gemini-1.5-flash-002',
            'models/gemini-1.5-flash-latest',
            'models/gemini-2.0-flash'
        ]
        for b in backups:
            if b not in candidate_models:
                candidate_models.append(b)

        last_error = None
        for m_name in candidate_models:
            try:
                model = genai.GenerativeModel(m_name, system_instruction=system_prompt)
                response = model.generate_content(query)
                return response.text
            except Exception as e:
                last_error = e
                continue
                
        return f"Gemini API Error: {str(last_error)}. Please check that your API key is valid."
    elif model_choice.lower() == "claude":
        client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": query}]
        )
        return response.content[0].text
    else:
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        )
        return response.choices[0].message.content
