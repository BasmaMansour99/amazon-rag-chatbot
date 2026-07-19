import pandas as pd
import numpy as np
import faiss
import re
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. Groq API Configuration (Llama 3.2 Cloud)
# ==========================================
GROQ_API_KEY = "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"

# ==========================================
# 2. Initialize RAG System & Clean Data
# ==========================================
@st.cache_resource
def init_rag_system():
    # Load embedding model
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Read the data file from the current directory
    DATA_PATH = "1429_1 - Copy.csv"
    REQUIRED_COLUMNS = [
        "id", "name", "asins", "brand", "categories",
        "reviews.didPurchase", "reviews.doRecommend",
        "reviews.rating", "reviews.title", "reviews.text", "reviews.username",
    ]
    raw_df = pd.read_csv(DATA_PATH, usecols=REQUIRED_COLUMNS)
    
    # Data Cleaning pipeline
    df = raw_df.copy()
    
    # Convert all columns to string and strip spaces to prevent unintended dropping
    for col in REQUIRED_COLUMNS:
        df[col] = df[col].astype(str).str.strip()
        
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.drop_duplicates(subset=["id", "reviews.username", "reviews.text"]).reset_index(drop=True)
    
    # Filter out rows where review text is actually empty or missing
    df = df[~df["reviews.text"].isin(["", "nan", "None", "none", "NaN"])].reset_index(drop=True)
    
    # Fill remaining missing fields with 'Unknown' instead of dropping the row
    for col in ["id", "asins", "name", "brand", "categories", "reviews.title", "reviews.username"]:
        df[col] = df[col].replace(["", "nan", "NaN", "None"], "Unknown")
        
    df["reviews.rating"] = pd.to_numeric(df["reviews.rating"], errors="coerce")
    df["reviews.rating"] = df["reviews.rating"].fillna(5.0)
    
    df["reviews.didPurchase"] = df["reviews.didPurchase"].replace(["True", "TRUE", "true", "1"], True)
    df["reviews.didPurchase"] = df["reviews.didPurchase"].replace(["False", "FALSE", "false", "0", "Unknown"], False).astype(bool)
    
    df["reviews.doRecommend"] = df["reviews.doRecommend"].replace(["True", "TRUE", "true", "1"], True)
    df["reviews.doRecommend"] = df["reviews.doRecommend"].replace(["False", "FALSE", "false", "0", "Unknown"], False).astype(bool)
    
    # Generate Embeddings & Build FAISS Index
    texts_to_embed = (df["reviews.title"] + ". " + df["reviews.text"]).tolist()
    embeddings = embedding_model.encode(texts_to_embed, batch_size=128, show_progress_bar=False)
    embeddings = np.array(embeddings).astype('float32')
    
    dimension = embeddings.shape[1]
    db = faiss.IndexFlatL2(dimension)
    db.add(embeddings)
    
    return embedding_model, db, df

# Launch system initialization
with st.spinner("Initializing Amazon Reviews Database... Please wait."):
    embedding_model, db, df = init_rag_system()

# ==========================================
# 3. RAG Core Engine & Generation
# ==========================================
def ask_llm(query, k=5):
    # Retrieval step
    query_vector = embedding_model.encode([query]).astype('float32')
    D, I = db.search(query_vector, k)
    retrieved_docs = df.iloc[I[0]]
    
    # Context Construction
    blocks = []
    for i, (_, row) in enumerate(retrieved_docs.iterrows(), start=1):
        blocks.append(
            f"[Source {i}] Product: {row['name']} | Brand: {row['brand']} | Rating: {row['reviews.rating']}/5\n"
            f"{row['reviews.title']}: {row['reviews.text']}"
        )
    context_text = "\n\n".join(blocks)
    
    if not context_text.strip():
        return "Insufficient information."
        
    # Strict Grounded Prompt Formulation
    prompt = f"""You are AutoAnalyst AI, an assistant that answers questions about
Amazon products using ONLY the customer review excerpts provided below.

Rules:
- Answer strictly using the information contained in the reviews below.
- Do not use outside knowledge or make assumptions beyond what is written.
- If the reviews do not contain enough information to answer the question,
  respond with exactly this sentence and nothing else:
  Insufficient information.
- Keep the answer brief and directly grounded in the review text.
- Where useful, mention which product(s) the answer is based on.

Question:
{query}

Customer review excerpts:
{context_text}

Answer:"""

    # Request payload for Groq Cloud (Llama 3.2 3B)
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.2-3b-preview",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0
    }
    
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            return "Insufficient information."
    except:
        return "Insufficient information."

# ==========================================
# 4. Streamlit Chat User Interface
# ==========================================
st.set_page_config(page_title="Amazon RAG Analyst", page_icon="🤖", layout="wide")
st.title("🤖 AutoAnalyst AI — Amazon Reviews Chat")
st.write("Ask questions about product features, quality, or customer feedback based on verified Amazon reviews.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the products/reviews in the dataset..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing customer reviews..."):
            answer = ask_llm(prompt)
            st.markdown(answer)
            
    st.session_state.messages.append({"role": "assistant", "content": answer})
