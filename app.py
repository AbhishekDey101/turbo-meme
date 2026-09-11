import streamlit as st
from datasets import load_dataset
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Setup the Webpage formatting
st.set_page_config(page_title="Abhishek Dey's Legal AI", page_icon="⚖️")
st.title("Abhishek Dey's Legal Precedent Engine")
st.markdown("Map complex factual scenarios to relevant Indian case law using semantic AI vector embeddings.")

# 2. Caching forces the server to load the heavy AI stuff only once, saving memory
@st.cache_resource
def load_ai_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_and_embed_data():
    # Fetch the token we just saved in Streamlit
    hf_token = st.secrets["HF_TOKEN"]
    
    # Load 2,500 cases to safely fit in the free 1GB RAM limit, using the token for access
    dataset = load_dataset("vaquill/open-india-law", "judgments", split="train[:2500]", token=hf_token)
    df = dataset.to_pandas()
    df['search_text'] = df['text'].fillna('')
    
    # Generate vectors
    model = load_ai_model()
    embeddings = model.encode(df['search_text'].tolist())
    return df, embeddings
    
# 3. Boot up the engine 
with st.spinner("Initializing AI Engine & Loading Precedent... (This takes a minute on first boot)"):
    model = load_ai_model()
    df, embeddings = load_and_embed_data()

st.success("Engine Ready.")

# 4. The Search Interface
query = st.text_area(
    "Enter Case Facts or Legal Issue",
    placeholder="E.g., Does the alienation of state-owned infrastructure to private monopolies violate the Public Trust Doctrine under Article 39(b) of the Constitution?",
    height=100
)

if st.button("Search Precedent", type="primary"):
    if query:
        with st.spinner("Searching semantic vectors..."):
            # Convert user query to vector and find matches
            query_embedding = model.encode([query])
            similarities = cosine_similarity(query_embedding, embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:3]
            
            st.markdown("### Top AI Matches")
            
            # Display results
            for idx in top_indices:
                match_score = round(similarities[idx] * 100, 2)
                case_id = df.iloc[idx].get('case_id', 'Citation Not Found')
                chunk_text = df.iloc[idx].get('text', '')[:750] + " [...]"
                
                st.markdown(f"**Match Confidence:** `{match_score}%` | **Case ID:** `{case_id}`")
                st.info(f"**Relevant Holding:**\n\n{chunk_text}")
                st.divider()
    else:
        st.warning("Please enter a case fact or legal issue to search.")
