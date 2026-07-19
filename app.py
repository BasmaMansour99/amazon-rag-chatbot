# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import faiss
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer

# ==========================================
# 0. Streamlit Page Configuration
# ==========================================
# Must be the very first Streamlit command called after imports
st.set_page_config(page_title="AR", layout="wide")

# ==========================================
# 1. Groq API Configuration
# ==========================================
# Fetch the secret API Key from Streamlit Secrets Management
if "GROQ_API_KEY" not in st.secrets:
    st.error("Configuration Error: 'GROQ_API_KEY' not found in Streamlit Secrets. Please configure your deployment application environment variables.")
    st.stop()

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# ==========================================
# 2. Chunking Helper Functions
# ==========================================
def chunk_text(text, chunk_size=38, overlap=10):
    """
    Splits input text into standardized token chunks based on explicit word counts.
    """
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
    """
    Validates files, processes input documents, indexes text structures using FAISS, 
    and handles downstream initialization component failures gracefully.
    """
    # Load embedding model with explicit error handling
    try:
        embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Initialization Error: Failed to load SentenceTransformer model. Details: {str(e)}")
        st.stop()

    # Read the data file from the current directory
    DATA_PATH = "1429_1 - Copy.csv"
    REQUIRED_COLUMNS = [
        "id", "name", "asins", "brand", "categories",
        "reviews.didPurchase", "reviews.doRecommend",
        "reviews.rating", "reviews.title", "reviews.text", "reviews.username",
    ]
    
    try:
        raw_df = pd.read_csv(DATA_PATH, usecols=REQUIRED_COLUMNS)
    except FileNotFoundError:
        st.error(f"Data Access Error: Target dataset file '{DATA_PATH}' could not be located in the working directory.")
        st.stop()
    except Exception as e:
        st.error(f"Data Processing Error: Unable to read target dataset file. Details: {str(e)}")
        st.stop()

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

    # Check if chunks were successfully generated
    if df_chunks.empty:
        st.error("Processing Error: No valid text records were found or generated after data cleaning.")
        st.stop()

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
    """
    Executes similarity matching against the FAISS matrix and structures context for LLM evaluation.
    """
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

    # Request payload configuration targeting Groq Llama-3.1-8b-instant
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0
    }

    try:
        # Implemented an extended 60-second execution window for Cloud environments
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions", 
            headers=headers, 
            json=payload, 
            timeout=60
        )
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            return f"Error from system API: HTTP {response.status_code} - {response.text}"
    except requests.exceptions.Timeout:
        return "Insufficient information. (System Connection Error: Remote API request timed out after 60 seconds.)"
    except requests.exceptions.RequestException as e:
        return f"Insufficient information. (System Connection Error: Network exception encountered. Details: {str(e)})"
    except Exception as e:
        return f"Insufficient information. (System Connection Error: {str(e)})"

# ==========================================
# 5. User Interface Configuration
# ==========================================
st.title("AR")

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
