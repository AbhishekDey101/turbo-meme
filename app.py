import streamlit as st
from datasets import load_dataset
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Setup the Webpage formatting
st.set_page_config(page_title="Abhishek Dey | Legal Precedent Engine", page_icon="⚖️", layout="wide")
st.title("Abhishek Dey's Legal Precedent & Regulatory Engine")
st.markdown("Semantic cross-referencing across apex court jurisprudence and financial market regulations.")

# 2. Load the AI Model (Cached to save memory)
@st.cache_resource
def load_ai_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# 3. Dynamic Data Loader (Streams based on the user's selected database)
@st.cache_data
def load_and_embed_data(database_choice):
    hf_token = st.secrets["HF_TOKEN"]
    model = load_ai_model()
    
    if database_choice == "Supreme Court Judgments":
        # Stream the judgments database
        dataset = load_dataset("vaquill/open-india-law", "judgments", split="train", streaming=True, token=hf_token)
        df = pd.DataFrame(list(dataset.take(1500)))
        df['search_text'] = df['text'].fillna('')
        
    elif database_choice == "SEBI Regulations":
        # Stream the specific SEBI regulations file
        dataset = load_dataset("vaquill/open-india-law", data_files="in_sebi_regulations.parquet", split="train", streaming=True, token=hf_token)
        df = pd.DataFrame(list(dataset.take(1500)))
        # SEBI data has 'title' and 'section_title' instead of just 'text'
        df['search_text'] = df['title'].fillna('') + " - " + df['section_title'].fillna('') + ": " + df['text'].fillna('')
        
    embeddings = model.encode(df['search_text'].tolist())
    return df, embeddings

# 4. The UI: Database Toggle
st.sidebar.header("Data Sources")
db_selection = st.sidebar.radio(
    "Select Legal Database:",
    ("Supreme Court Judgments", "SEBI Regulations")
)

st.sidebar.info("This MVP limits the database to 1,500 semantic chunks per source to optimize for free cloud hosting limits.")

# 5. Boot up the selected engine 
with st.spinner(f"Initializing AI Engine & Loading {db_selection}..."):
    model = load_ai_model()
    df, embeddings = load_and_embed_data(db_selection)

st.success(f"Engine Ready: Connected to {db_selection}")

# 6. The Search Interface
query = st.text_area(
    "Enter Case Facts, Regulatory Issue, or Legal Concept",
    placeholder="E.g., Does the alienation of state-owned infrastructure to private monopolies violate the Public Trust Doctrine under Article 39(b)?",
    height=100
)

if st.button("Search Database", type="primary"):
    if query:
        with st.spinner("Searching semantic vectors..."):
            query_embedding = model.encode([query])
            similarities = cosine_similarity(query_embedding, embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:3]
            
            st.markdown(f"### Top AI Matches in {db_selection}")
            
            for idx in top_indices:
                match_score = round(similarities[idx] * 100, 2)
                
                # Format output differently depending on the database schema
                if db_selection == "Supreme Court Judgments":
                    doc_id = df.iloc[idx].get('case_id', 'Citation Not Found')
                    chunk_text = df.iloc[idx].get('text', '')[:750] + " [...]"
                    st.markdown(f"**Match Confidence:** `{match_score}%` | **Case ID:** `{doc_id}`")
                    st.info(f"**Relevant Holding:**\n\n{chunk_text}")
                
                elif db_selection == "SEBI Regulations":
                    doc_id = df.iloc[idx].get('title', 'Unknown Act')
                    section = df.iloc[idx].get('section_title', 'Unknown Section')
                    chunk_text = df.iloc[idx].get('text', '')[:750] + " [...]"
                    st.markdown(f"**Match Confidence:** `{match_score}%` | **Regulation:** `{doc_id}`")
                    st.info(f"**Section: {section}**\n\n{chunk_text}")
                    
                st.divider()
    else:
        st.warning("Please enter a legal issue to search.")
