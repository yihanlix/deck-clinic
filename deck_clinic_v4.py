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
    page_title="Deck Clinic V6: Narrative Engine",
    page_icon="💾",
    layout="wide"
)

# --- 2. CSS STYLING (Retro + Clinic Cards) ---
st.markdown("""
<style>
    /* A. Retro Typewriter Font */
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&display=swap');
    
    * {
        font-family: 'Space Mono', monospace;
    }
    
    /* B. Clean Headers */
    h1, h2, h3, h4, h5 {
        font-weight: 700;
        letter-spacing: -1px;
    }
    
    /* C. Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        background-color: #f0f2f6;
        padding: 5px 10px;
        border-radius: 5px;
        border-left: 5px solid #ff4b4b;
    }
    
    /* D. Buttons */
    div.stButton > button {
        border-radius: 0px;
        border: 2px solid #000;
        font-weight: bold;
        box-shadow: 2px 2px 0px #000;
        transition: all 0.1s;
    }
    div.stButton > button:hover {
        box-shadow: 4px 4px 0px #000;
        transform: translate(-1px, -1px);
    }
    
    /* E. CLINIC CARD DESIGN (Deep Dive) */
    .issue-tag {
        background-color: #ffebee; /* Light Red */
        color: #c62828;
        padding: 5px 10px;
        font-weight: bold;
        border-left: 4px solid #c62828;
        margin-bottom: 8px;
        font-size: 0.85rem;
    }
    .fix-tag {
        background-color: #e8f5e9; /* Light Green */
        color: #2e7d32;
        padding: 5px 10px;
        font-weight: bold;
        border-left: 4px solid #2e7d32;
        margin-bottom: 8px;
        font-size: 0.85rem;
    }
    .logic-footer {
        font-size: 0.85rem;
        color: #555;
        background-color: #fafafa;
        padding: 10px;
        border-top: 1px dashed #ccc;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. SECURITY & SETUP ---
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    st.error("🚨 SYSTEM ERROR: API Key Missing.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key
genai.configure(api_key=api_key)

# --- 4. CORE ENGINE ---
@st.cache_resource
def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

embeddings = get_embedding_model()
PERSIST_DIRECTORY = "deck_memory_db"

# --- 5. SIDEBAR: CONTROL PANEL ---
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

# --- 6. MAIN INTERFACE ---
st.title("💾 DECK CLINIC V6")
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

    # --- V6 PROMPT STRUCTURE ---
    prompt = f"""
    {base_instruction}
    
    ### GOLD STANDARD CONTEXT:
    {knowledge_context}
    
    ### DRAFT TEXT:
    {draft_text[:50000]} 
    
    ### INSTRUCTIONS:
    1. **STEP 1 (HIDDEN BRAINSTORM):** Read the text. Specifically look for logical gaps. Ask yourself: "Does the problem prove the solution?" "Is the data specific?"
    2. **STEP 2 (SCORING):** Only assign scores AFTER you have written the critique.
    3. **STEP 3 (EXTRACTION):** Extract the current headlines to identify the existing narrative.
    4. **STEP 4 (Headline & Narrative Audit):**
       - Critique the current headlines: Do they tell a story if read in isolation? Are they descriptive or generic?
       - Suggest a **"Revised Headline Flow"**: A list of rewritten headlines that guide the reader logically from the problem to the solution.
    
    ### JSON STRUCTURE:
    {{
        "reasoning_log": "<string: Write a 3-sentence internal analysis of the logic flaws here FIRST.>",
        "scores": {{
            "Logic": <int 0-100>,
            "Clarity": <int 0-100>,
            "Impact": <int 0-100>
        }},
        "executive_summary": "<string: Brutal one-sentence summary based on the reasoning_log>",
        "narrative_check": {{
             "original_headlines": [
                 "<string: Extracted Headline 1>",
                 "<string: Extracted Headline 2>"
             ],
             "critique": "<string: Critique of the current storytelling flow>",
             "revised_headlines": [
                 "<string: Improved Headline 1>",
                 "<string: Improved Headline 2>"
             ]
        }},
        "section_deep_dive": [
            {{
                "target_section": "<string: Name of original section>",
                "issue": "<string: Specific critique (e.g. Logic gap, Passive voice)>",
                "improved_version": "<string: The rewritten text>",
                "why": "<string: Why this is better>"
            }}
        ]
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
            s1.metric("LOGIC", f"{data.get('scores', {}).get('Logic', 0)}")
            s2.metric("CLARITY", f"{data.get('scores', {}).get('Clarity', 0)}")
            s3.metric("IMPACT", f"{data.get('scores', {}).get('Impact', 0)}")
        
        st.divider()
        st.info(f"**EXECUTIVE SUMMARY:** {data.get('executive_summary', 'No summary generated.')}")
        
        # --- NEW TABS LAYOUT ---
        tab1, tab2 = st.tabs(["STORY FLOW", "🔬 DEEP DIVE & REWRITES"])
        
        # --- TAB 1: NARRATIVE FLOW ---
        with tab1:
            st.markdown("#### The Narrative Check (Pyramid Principle)")
            nav_data = data.get('narrative_check', {})
            
            st.markdown(f"> *{nav_data.get('critique', 'No critique available.')}*")
            st.divider()

            col_a, col_b = st.columns(2)
            with col_a:
                st.caption("🔴 ORIGINAL FLOW")
                for line in nav_data.get('original_headlines', []):
                    st.text(f"• {line}")
            
            with col_b:
                st.caption("🟢 OPTIMIZED FLOW")
                for line in nav_data.get('revised_headlines', []):
                    st.markdown(f"**• {line}**")

            if data.get('scores', {}).get('Logic', 0) < 75:
                st.error("⚠️ NARRATIVE THREAD BROKEN")
            else:
                st.success("✅ NARRATIVE THREAD STABLE")
        
        # --- TAB 2: SECTIONS DEEP DIVE (SPLIT VIEW) ---
        with tab2:
            st.markdown("#### 🔬 Surgical Reconstruction")
            st.caption("Specific text edits to improve Logic, Clarity, and Impact.")
            
            deep_dive_items = data.get('section_deep_dive', [])
            
            if not deep_dive_items:
                st.info("✅ No critical issues found. Your deck is clean!")
            
            for i, item in enumerate(deep_dive_items):
                # Container for visual grouping
                with st.container():
                    st.markdown(f"##### 📍 Section: {item.get('target_section', 'General')}")
                    
                    # Layout: 1/