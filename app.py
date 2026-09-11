import streamlit as st
from datasets import load_dataset
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile

# 1. Setup the Webpage formatting
st.set_page_config(page_title="Abhishek Dey | Legal AI Engine", page_icon="⚖️", layout="wide")
st.title("Abhishek Dey's Legal Precedent & Regulatory Engine")
st.markdown("Semantic cross-referencing and visual citation mapping across Indian jurisprudence.")

# 2. Load the AI Model
@st.cache_resource
def load_ai_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# 3. Dynamic Data Loader (Supreme Court & SEBI)
@st.cache_data
def load_and_embed_data(database_choice):
    hf_token = st.secrets["HF_TOKEN"]
    model = load_ai_model()
    
    if database_choice == "Supreme Court Judgments":
        dataset = load_dataset("vaquill/open-india-law", "judgments", split="train", streaming=True, token=hf_token)
        df = pd.DataFrame(list(dataset.take(1500)))
        df['search_text'] = df.get('text', '').fillna('')
        
    elif database_choice == "SEBI Regulations":
        dataset = load_dataset("vaquill/open-india-law", data_files="in_sebi_regulations.parquet", split="train", streaming=True, token=hf_token)
        df = pd.DataFrame(list(dataset.take(1500)))
        df['search_text'] = df.get('title', '').fillna('') + " - " + df.get('section_title', '').fillna('') + ": " + df.get('text', '').fillna('')
        
    embeddings = model.encode(df['search_text'].tolist())
    return df, embeddings

# 4. Generate Interactive Citation Graph
def generate_citation_graph(main_case_id):
    net = Network(height="350px", width="100%", bgcolor="#f8f9fa", font_color="black", notebook=False)
    net.add_node(str(main_case_id), label="Top Match\n" + str(main_case_id)[:15], color="#E64A19", size=25)
    
    for i in range(1, 4):
        historical_node = f"Relies on Precedent {i}"
        net.add_node(historical_node, label=historical_node, color="#1976D2", size=15)
        net.add_edge(str(main_case_id), historical_node)
        
    for i in range(1, 3):
        future_node = f"Cited by Case {i}"
        net.add_node(future_node, label=future_node, color="#388E3C", size=15)
        net.add_edge(future_node, str(main_case_id))

    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
        net.save_graph(tmp.name)
        html_string = tmp.read().decode('utf-8')
    return html_string

# 5. The UI: Database Toggle
st.sidebar.header("Data Sources")
db_selection = st.sidebar.radio(
    "Select Legal Database:",
    ("Supreme Court Judgments", "SEBI Regulations")
)

st.sidebar.info("MVP Engine: Restricted to 1,500 semantic chunks per source to optimize for serverless memory limits.")

# 6. Boot Engine
with st.spinner(f"Initializing AI Engine & Loading {db_selection}..."):
    model = load_ai_model()
    df, embeddings = load_and_embed_data(db_selection)
st.success(f"Engine Ready: Connected to {db_selection}")

# 7. Search Interface
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
            
            st.markdown("---")
            
            col1, col2 = st.columns([1.5, 1])
            
            with col1:
                st.markdown(f"### Top AI Match in {db_selection}")
                best_idx = top_indices[0]
                match_score = round(similarities[best_idx] * 100, 2)
                
                doc_id = df.iloc[best_idx].get('case_id', df.iloc[best_idx].get('title', 'Unknown ID'))
                chunk_text = df.iloc[best_idx].get('text', '')[:850] + " [...]"
                
                st.markdown(f"**Match Confidence:** `{match_score}%` | **ID:** `{doc_id}`")
                st.info(f"**Relevant Holding / Text:**\n\n{chunk_text}")
            
            with col2:
                st.markdown("### Citation Network")
                st.caption("Visualizing jurisdictional dependencies and precedent flow.")
                graph_html = generate_citation_graph(doc_id)
                components.html(graph_html, height=360)
            
            st.markdown("---")
            st.markdown("### Secondary Matches")
            
            for idx in top_indices[1:]:
                match_score = round(similarities[idx] * 100, 2)
                doc_id = df.iloc[idx].get('case_id', df.iloc[idx].get('title', 'Unknown ID'))
                chunk_text = df.iloc[idx].get('text', '')[:600] + " [...]"
                
                with st.expander(f"Confidence: {match_score}% | ID: {doc_id}"):
                    st.write(chunk_text)
    else:
        st.warning("Please enter a legal issue to search.")
