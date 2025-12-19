import streamlit as st
import google.generativeai as genai
import os
import tempfile
import json
import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Deck Clinic V5: Retro Lab",
    page_icon="💾",
    layout="wide"
)

# --- RETRO LAB CSS (Stable & Clean) ---
st.markdown("""
<style>
    /* 1. Retro Typewriter Font */
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&display=swap');
    
    * {
        font-family: 'Space Mono', monospace;
    }
    
    /* 2. Clean Headers */
    h1, h2, h3 {
        font-weight: 700;
        letter-spacing: -1px;
    }
    
    /* 3. Metric Cards (Simple Box) */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        background-color: #f0f2f6;
        padding: 5px 10px;
        border-radius: 5px;
        border-left: 5px solid #ff4b4b; /* Streamlit Red Accent */
    }
    
    /* 4. Expander Borders */
    .stExpander {
        border: 2px solid #000;
        border-radius: 0px;
    }
    
    /* 5. Buttons (Retro Rectangles) */
    div.stButton > button {
        border-radius: 0px;
        border: 2px solid #000;
        font-weight: bold;
        box-shadow: 2px 2px 0px #000; /* Drop shadow effect */
        transition: all 0.1s;
    }
    div.stButton > button:hover {
        box-shadow: 4px 4px 0px #000;
        transform: translate(-1px, -1px);
    }
</style>
""", unsafe_allow_html=True)

# --- 2. SECURITY & SETUP ---
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    st.error("🚨 SYSTEM ERROR: API Key Missing.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key
genai.configure(api_key=api_key)

# --- 3. CORE ENGINE ---
@st.cache_resource
def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

embeddings = get_embedding_model()
PERSIST_DIRECTORY = "deck_memory_db"

# --- 4. SIDEBAR: CONTROL PANEL ---
with st.sidebar:
    st.title("🎛️ SETTINGS")
    
    # Context Selector
    doc_type = st.selectbox(
        "DIAGNOSTIC PROTOCOL",
        ["Strategy Deck (McKinsey/Amazon)", "Product Spec (Technical)", "Exec Update (Brief)"]
    )
    
    st.divider()
    
    # Knowledge Base Uploader
    st.caption("📂 KNOWLEDGE BASE")
    uploaded_file = st.file_uploader("Upload 'Gold Standard' PDF", type="pdf")
    
    if uploaded_file and st.button("TRAIN SYSTEM"):
        with st.spinner("Indexing..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            loader = PyPDFLoader(tmp_file_path)
            raw_docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=500)
            docs = text_splitter.split_documents(raw_docs)
            
            vector_db = Chroma.from_documents(docs, embeddings, persist_directory=PERSIST_DIRECTORY)
            try: vector_db.persist()
            except: pass
            st.success(f"System Index Updated: {len(docs)} chunks.")

# --- 5. MAIN INTERFACE ---
st.title("💾 DECK CLINIC V5")
st.caption(f"PROTOCOL: {doc_type} | CORE: gemini-flash-latest") 

col1, col2 = st.columns([2, 3]) 

with col1:
    st.markdown("### INPUT FEED")
    target_pdf = st.file_uploader("Upload Draft PDF", type="pdf", key="target")
    analyze_btn = st.button("RUN DIAGNOSTIC", type="primary", use_container_width=True)

if target_pdf and analyze_btn:
    # A. File Processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(target_pdf.read())
        draft_path = tmp_file.name
    
    loader = PyPDFLoader(draft_path)
    draft_docs = loader.load()
    draft_text = " ".join([d.page_content for d in draft_docs])

    # B. RAG Retrieval
    with st.spinner("Retrieving Context..."):
        try:
            vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
            results = vector_db.similarity_search(draft_text, k=3)
            knowledge_context = "\n".join([doc.page_content for doc in results])
        except:
            knowledge_context = "Standard Top Tech Company Protocols"

    # --- C. DYNAMIC PROMPT LOGIC ---
    base_instruction = ""
    
    if "Strategy" in doc_type:
        base_instruction = """
        ROLE: VP of Strategy (Amazon/McKinsey background).
        FRAMEWORK:
        1. **Amazon Clarity:** Grade 8 reading level. No big words. No long sentences (>2 commas).
        2. **McKinsey Structure:** MECE principle. Pyramid Principle. "Golden Thread" logic.
        """
    elif "Product" in doc_type:
        base_instruction = """
        ROLE: Senior Technical PM.
        FRAMEWORK:
        1. **Feasibility:** Identify missing edge cases or technical risks.
        2. **Specs:** Ensure metrics and success criteria are defined strictly.
        """
    else: # Exec Update
        base_instruction = """
        ROLE: CEO.
        FRAMEWORK:
        1. **BLUF:** Bottom Line Up Front.
        2. **Brevity:** If it can be said in 5 words, do not use 10.
        """

    prompt = f"""
    {base_instruction}
    
    ### GOLD STANDARD CONTEXT:
    {knowledge_context}
    
    ### DRAFT TEXT:
    {draft_text[:50000]} 
    
    ### INSTRUCTIONS:
    1. Analyze the text based on the FRAMEWORK above.
    2. **CRITICAL STEP:** Extract the 'Narrative Flow'. Identify the Headline/Topic Sentence of every distinct section. List them in order to test the story flow.
    3. Output in JSON only.

    ### JSON STRUCTURE:
    {{
        "scores": {{
            "Logic": <int 0-100>,
            "Clarity": <int 0-100>,
            "Impact": <int 0-100>
        }},
        "executive_summary": "<string: Brutal one-sentence summary>",
        "narrative_check": [
             "<string: Slide/Section 1 Headline>",
             "<string: Slide/Section 2 Headline>",
             "<string: Slide/Section 3 Headline>"
        ],
        "critical_issues": [
            {{ "section": "<string>", "issue": "<string>", "fix": "<string>" }}
        ],
        "rewrite_showcase": {{
            "original_text": "<string>",
            "improved_version": "<string>",
            "why": "<string>"
        }}
    }}
    """

    # D. Generation
    with st.spinner("Processing Logic Matrix..."):
        model = genai.GenerativeModel('gemini-flash-latest')
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        
    # E. Rendering
    try:
        data = json.loads(response.text)
        
        with col2:
            st.markdown("### SCORECARD")
            s1, s2, s3 = st.columns(3)
            s1.metric("LOGIC", f"{data['scores'].get('Logic',0)}")
            s2.metric("CLARITY", f"{data['scores'].get('Clarity',0)}")
            s3.metric("IMPACT", f"{data['scores'].get('Impact',0)}")
        
        st.divider()
        
        # --- TABS LAYOUT ---
        tab1, tab2, tab3 = st.tabs(["STORY FLOW", "CRITICAL GAPS", "REWRITE LAB"])
        
        with tab1:
            st.markdown("#### The Narrative Check (Pyramid Principle)")
            st.caption("Does the story hold together reading ONLY these lines?")
            with st.container(border=True):
                for i, line in enumerate(data.get('narrative_check', [])):
                    st.code(f"{i+1}. {line}", language="markdown")
            
            if data['scores'].get('Logic', 0) < 75:
                st.error("⚠️ NARRATIVE THREAD BROKEN")
            else:
                st.success("✅ NARRATIVE THREAD STABLE")
        
        with tab2:
            st.info(f"**SUMMARY:** {data['executive_summary']}")
            for item in data['critical_issues']:
                with st.expander(f"📍 {item['section']}"):
                    st.write(f"**ISSUE:** {item['issue']}")
                    st.markdown(f"**FIX:** `{item['fix']}`")
        
        with tab3:
            c1, c2 = st.columns(2)
            with c1:
                st.caption("ORIGINAL")
                st.code(data['rewrite_showcase']['original_text'], language="text")
            with c2:
                st.caption("OPTIMIZED")
                st.code(data['rewrite_showcase']['improved_version'], language="text")
            st.markdown(f"> **LOGIC:** {data['rewrite_showcase']['why']}")

    except Exception as e:
        st.error(f"Data Stream Parsing Error: {e}")
        with st.expander("DEBUG DATA"):
            st.text(response.text)