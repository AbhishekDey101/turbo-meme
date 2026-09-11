import streamlit as st
from datasets import load_dataset
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
import PyPDF2
import google.generativeai as genai
import os

# 1. Setup the Webpage formatting
st.set_page_config(page_title="Abhishek Dey | Legal AI Suite", page_icon="⚖️", layout="wide")
st.title("Abhishek Dey's Advanced Legal AI Suite")

# 2. Initialize Models and API
@st.cache_resource
def load_semantic_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def configure_genai():
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        return genai.GenerativeModel('gemini-1.5-flash')
    except:
        return None

# 3. Dynamic Data Loader
@st.cache_data
def load_and_embed_data(database_choice):
    hf_token = st.secrets["HF_TOKEN"]
    model = load_semantic_model()
    
   # ADDED: Calcutta and Telangana High Court parquet files
    file_map = {
        "Supreme Court Judgments": {"split": "train", "data_files": None},
        "Calcutta High Court": {"split": "train", "data_files": "in_calcutta_judgments.parquet"},
        "Telangana High Court": {"split": "train", "data_files": "in_telangana_judgments.parquet"},
        "SEBI (Securities)": {"split": "train", "data_files": "in_sebi_regulations.parquet"},
        "MCA (Corporate Affairs)": {"split": "train", "data_files": "in_mca_regulations.parquet"}
    }
    config = file_map[database_choice]
    
    if config["data_files"] is None:
        dataset = load_dataset("vaquill/open-india-law", "judgments", split=config["split"], streaming=True, token=hf_token)
    else:
        dataset = load_dataset("vaquill/open-india-law", data_files=config["data_files"], split=config["split"], streaming=True, token=hf_token)
        
    df = pd.DataFrame(list(dataset.take(1500)))
    
    if 'section_title' in df.columns:
        df['search_text'] = df.get('title', '') + " - " + df.get('section_title', '') + ": " + df.get('text', '')
    else:
        df['search_text'] = df.get('text', '')
        
    embeddings = model.encode(df['search_text'].tolist())
    return df, embeddings

# 4. Sidebar Configuration
st.sidebar.header("Global Settings")
db_selection = st.sidebar.radio(
    "Active Database:", 
    (
        "Supreme Court Judgments", 
        "Calcutta High Court",
        "Telangana High Court",
        "SEBI (Securities)", 
        "MCA (Corporate Affairs)"
    )
)
filter_repealed = st.sidebar.checkbox("Filter out Repealed/Overruled (Status Checker)", value=True)
with st.spinner(f"Mounting {db_selection} into memory..."):
    semantic_model = load_semantic_model()
    df, embeddings = load_and_embed_data(db_selection)
    llm = configure_genai()

# 5. Build the UI Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Research & Synthesis", "📄 Draft Analyzer", "🛡️ Counter-Argument Engine"])

# ==========================================
# TAB 1: RESEARCH & SYNTHESIS (Features 2 & 4)
# ==========================================
with tab1:
    st.markdown("### Semantic Search & AI Synthesis")
    query = st.text_area("Enter Legal Issue", placeholder="E.g., Anticipatory bail criteria for non-executive directors under PMLA.", height=100)
    
    if st.button("Search & Synthesize", type="primary"):
        if query:
            with st.spinner("Analyzing semantic vectors..."):
                query_embedding = semantic_model.encode([query])
                similarities = cosine_similarity(query_embedding, embeddings)[0]
                
                # Feature 4: Status Checker Filtering
                valid_indices = []
                for idx in np.argsort(similarities)[::-1]:
                    # If filter is on, skip rows explicitly marked as repealed or spent
                    if filter_repealed and 'act_status' in df.columns:
                        status = str(df.iloc[idx].get('act_status', '')).lower()
                        if 'repealed' in status or 'spent' in status:
                            continue
                    valid_indices.append(idx)
                    if len(valid_indices) == 3: break
                
                retrieved_texts = []
                st.markdown("---")
                st.markdown("### Top AI Matches")
                
                for idx in valid_indices:
                    match_score = round(similarities[idx] * 100, 2)
                    text = df.iloc[idx].get('text', '')
                    retrieved_texts.append(text)
                    st.info(f"**Match: {match_score}%**\n\n{text[:500]}...")

                # Feature 2: Generative AI APA Synthesis
                if llm:
                    st.markdown("### 🤖 Generative AI Synthesis")
                    with st.spinner("Drafting APA-formatted research summary..."):
                        context = "\n\n".join(retrieved_texts)
                        prompt = f"""
                        Act as an expert legal academic. Synthesize the following retrieved Indian legal text into a cohesive research summary answering this query: "{query}".
                        Strictly adhere to APA citation guidelines for any references. Maintain a scholarly, objective tone suitable for a master's level academic research paper.
                        
                        Retrieved Law:
                        {context}
                        """
                        response = llm.generate_content(prompt)
                        st.success(response.text)
                else:
                    st.warning("Gemini API Key missing. Add it to Streamlit Secrets to enable GenAI.")

# ==========================================
# TAB 2: THE DRAFT ANALYZER (Feature 1)
# ==========================================
with tab2:
    st.markdown("### PDF Draft Analyzer")
    st.caption("Upload a moot court memorial, representation letter, or plaint. The AI will cross-reference your arguments against the database.")
    uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Analyze Draft"):
            with st.spinner("Extracting text and mapping arguments..."):
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                draft_text = " ".join([page.extract_text() for page in pdf_reader.pages])
                
                # Embed the first 2000 characters of the draft to find the core legal theme
                query_embedding = semantic_model.encode([draft_text[:2000]])
                similarities = cosine_similarity(query_embedding, embeddings)[0]
                best_idx = np.argsort(similarities)[::-1][0]
                
                st.markdown("### Draft Cross-Reference Results")
                st.write("**Identified Core Theme:** The AI mapped your document to the following authoritative precedent/regulation:")
                st.info(df.iloc[best_idx].get('text', '')[:1000] + "...")
                
                if llm:
                    with st.spinner("Generating critique..."):
                        prompt = f"""
                        You are a senior advocate reviewing a junior's draft. Compare the draft excerpt against the established precedent. 
                        Identify areas where the draft is strong, and flag areas where it contradicts the precedent.
                        Draft Excerpt: {draft_text[:1500]}
                        Precedent: {df.iloc[best_idx].get('text', '')}
                        """
                        response = llm.generate_content(prompt)
                        st.warning("**AI Critique & Fortification Suggestions:**\n\n" + response.text)

# ==========================================
# TAB 3: COUNTER-ARGUMENT ENGINE (Feature 3)
# ==========================================
with tab3:
    st.markdown("### Opposing Counsel / Counter-Argument Mode")
    st.caption("Enter a legal argument. The AI will hunt for exceptions, statutory defenses, and opposing interpretations.")
    claim = st.text_area("Enter your primary argument", placeholder="E.g., Mandatory pre-institution mediation under the Commercial Courts Act is an absolute bar to all litigation.", height=100)
    
    if st.button("Find Counter-Arguments", type="primary"):
        if claim and llm:
            with st.spinner("Hunting for exceptions in the database..."):
                # Search for the concept
                query_embedding = semantic_model.encode([claim])
                similarities = cosine_similarity(query_embedding, embeddings)[0]
                top_indices = np.argsort(similarities)[::-1][:5]
                
                context = "\n".join([df.iloc[idx].get('text', '') for idx in top_indices])
                
                prompt = f"""
                You are opposing counsel. The plaintiff is arguing: "{claim}".
                Based ONLY on the following Indian legal texts, formulate a counter-argument. Highlight any statutory exceptions, judicial discretion, or dissenting views that defeat the plaintiff's claim.
                Legal Texts:
                {context}
                """
                response = llm.generate_content(prompt)
                
                st.error("### 🛡️ The Counter-Argument")
                st.write(response.text)
        elif not llm:
             st.warning("Gemini API Key missing.")
