# -*- coding: utf-8 -*-
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
# Security Note: It is recommended to use st.secrets["GROQ_API_KEY"] instead of hardcoding it here.
# Make sure to replace this with your actual valid API key.
GROQ_API_KEY = "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" 

# ==========================================
# 2. Chunking Helper Functions
# ==========================================
def chunk_text(text, chunk_size=38, overlap=10):
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks

# ==========================================
# 3. Initialize Search System & Clean Data
# ==========================================
@st.cache_resource
def init_search_system():
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
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.drop_duplicates(subset=["id", "reviews.username", "reviews.text"]).reset_index(drop=True)
    df["reviews.text"] = df["reviews.text"].astype(str).str.strip()
    df = df[~df["reviews.text"].isin(["", "nan", "None", "none", "NaN"])].reset_index(drop=True)

    for col in ["name", "brand", "categories", "reviews.title", "reviews.username"]:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()
        df.loc[df[col] == "", col] = "Unknown"

    df["reviews.rating"] = pd.to_numeric(df["reviews.rating"], errors="coerce")
    df["reviews.rating"] = df["reviews.rating"].fillna(df["reviews.rating"].median())

    df["reviews.didPurchase"] = df["reviews.didPurchase"].fillna(False).astype(bool)
    df["reviews.doRecommend"] = df["reviews.doRecommend"].fillna(False).astype(bool)
    df = df.dropna(subset=["id", "asins"]).reset_index(drop=True)

    # === Apply Text Chunking for Vector Database Building ===
    chunked_records = []
    for idx, row in df.iterrows():
        full_text = f"{row['reviews.title']}. {row['reviews.text']}".strip()
        chunks = chunk_text(full_text, chunk_size=38, overlap=10)
        for chunk in chunks:
            chunked_records.append({
                "chunk_text": chunk,
                "name": row["name"],
                "brand": row["brand"],
                "rating": row["reviews.rating"],
                "title": row["reviews.title"]
            })
    
    # Convert chunked data into a new DataFrame
    df_chunks = pd.DataFrame(chunked_records)

    # Generate Embeddings for the text chunks
    texts_to_embed = df_chunks["chunk_text"].tolist()
    embeddings = embedding_model.encode(texts_to_embed, batch_size=256, show_progress_bar=False)
    embeddings = np.array(embeddings).astype('float32')

    dimension = embeddings.shape[1]
    db = faiss.IndexFlatL2(dimension)
    db.add(embeddings)

    return embedding_model, db, df_chunks

# Launch system initialization
with st.spinner("Initializing Amazon Reviews Database... Please wait."):
    embedding_model, db, df_chunks = init_search_system()

# ==========================================
# 4. Search Engine & Content Generation
# ==========================================
def query_database(query, k=4): 
    # Retrieval step
    query_vector = embedding_model.encode([query]).astype('float32')
    D, I = db.search(query_vector, k)
    
    # Get the corresponding chunks from the dataset
    retrieved_chunks = df_chunks.iloc[I[0]]

    # Context Construction
    blocks = []
    for i, (_, row) in enumerate(retrieved_chunks.iterrows(), start=1):
        blocks.append(
            f"[Source {i}] Product: {row['name']} | Brand: {row['brand']} | Rating: {row['rating']}/5\n"
            f"{row['chunk_text']}"
        )
    context_text = "\n\n".join(blocks)

    if not context_text.strip():
        return "Insufficient information."

    # Strict Grounded Prompt Formulation (No AI branding)
    prompt = f"""Answer the question using ONLY the customer review excerpts provided below.

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

    # Request payload for Groq Cloud
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
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            # Temporary error reporting for connection status
            return f"Error from system API: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Insufficient information. (System Connection Error: {str(e)})"

# ==========================================
# 5. User Interface Configuration
# ==========================================
st.set_page_config(page_title="Amazon Reviews System", layout="wide")
st.title("Amazon Product Reviews Search Engine")
st.write("Extract information regarding product features, quality, or customer feedback directly from database records.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter search query or question about dataset products..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processing database records..."):
            answer = query_database(prompt)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
