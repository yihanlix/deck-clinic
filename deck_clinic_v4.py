import streamlit as st
import google.generativeai as genai
import os
import tempfile
import json
import pandas as pd
import csv
import datetime
import uuid
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from pdf2image import convert_from_path 

# --- HELPER FUNCTIONS ---
def log_feedback(session_id, rating, comment, doc_type):
    feedback_file = "feedback_logs.csv"
    file_exists = os.path.isfile(feedback_file)
    with open(feedback_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Session ID", "Rating", "Comment", "Doc Type"])
        writer.writerow([datetime.datetime.now(), session_id, rating, comment, doc_type])

def log_interaction(session_id, filename, doc_type, scores, exec_summary):
    log_file = "clinic_logs.csv"
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Session ID", "Filename", "Doc Type", "Logic Score", "Clarity Score", "Impact Score", "Summary"])
        writer.writerow([
            datetime.datetime.now(), session_id, filename, doc_type,
            scores.get('Logic', 0), scores.get('Clarity', 0), scores.get('Impact', 0),
            exec_summary
        ])

def get_score_context(score):
    """Return tier, color, and advice based on score"""
    if score >= 85:
        return "EXCELLENT", "#2dd4bf", "Deck is presentation-ready"
    elif score >= 70:
        return "GOOD", "#60a5fa", "Minor refinements suggested"
    elif score >= 50:
        return "NEEDS WORK", "#fb923c", "Significant improvements needed"
    else:
        return "CRITICAL", "#f87171", "Major structural issues detected"

# --- ADVANCED RAG FUNCTIONS ---
def advanced_rag_pipeline(draft_text, doc_type, vector_db):
    """
    Advanced RAG with query expansion and reciprocal rank fusion
    """
    
    # STEP 1: Query Expansion
    query_expansion_prompt = f"""
    Generate 5 focused search queries to find relevant examples for this {doc_type}:
    
    DRAFT EXCERPT:
    {draft_text[:1500]}
    
    Return JSON array: ["query1", "query2", "query3", "query4", "query5"]
    Each query should focus on a different aspect (problem, solution, metrics, narrative, audience).
    """
    
    model = genai.GenerativeModel('gemini-flash-latest')
    try:
        response = model.generate_content(
            query_expansion_prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        queries = json.loads(response.text)
    except:
        # Fallback to simple query if expansion fails
        queries = [draft_text[:500]]
    
    # STEP 2: Multi-Query Retrieval with RRF
    all_results = {}
    
    for i, query in enumerate(queries):
        try:
            results = vector_db.similarity_search(query, k=5)
            
            for rank, doc in enumerate(results):
                doc_id = hash(doc.page_content[:200])
                if doc_id not in all_results:
                    all_results[doc_id] = {'doc': doc, 'ranks': []}
                all_results[doc_id]['ranks'].append(rank + 1)
        except:
            continue
    
    # STEP 3: Reciprocal Rank Fusion
    k = 60
    for doc_id in all_results:
        ranks = all_results[doc_id]['ranks']
        rrf_score = sum(1 / (k + r) for r in ranks)
        all_results[doc_id]['rrf_score'] = rrf_score
    
    sorted_docs = sorted(
        all_results.values(),
        key=lambda x: x['rrf_score'],
        reverse=True
    )
    
    final_docs = [item['doc'] for item in sorted_docs[:5]]
    
    # STEP 4: Assemble Context with Source Attribution
    knowledge_context = ""
    for i, doc in enumerate(final_docs, 1):
        knowledge_context += f"\n\n━━━ EXAMPLE {i} ━━━\n{doc.page_content}"
    
    return knowledge_context, queries, final_docs

# --- ENHANCED DOCUMENT INDEXING ---
def enhanced_document_indexing(uploaded_file):
    """
    Index documents with metadata extraction
    """
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name
    
    loader = PyPDFLoader(tmp_file_path)
    raw_docs = loader.load()
    
    # Extract metadata
    sample_text = "\n".join([doc.page_content for doc in raw_docs[:3]])
    
    metadata_prompt = f"""
    Analyze this document briefly:
    
    {sample_text[:2000]}
    
    Return JSON:
    {{
        "doc_type": "Strategy Deck | Product Spec | Fundraising Pitch",
        "quality_tier": "Excellent | Good | Average"
    }}
    """
    
    try:
        model = genai.GenerativeModel('gemini-flash-latest')
        response = model.generate_content(
            metadata_prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        metadata = json.loads(response.text)
    except:
        metadata = {"doc_type": "Unknown", "quality_tier": "Unknown"}
    
    # Split with metadata
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=500)
    docs = text_splitter.split_documents(raw_docs)
    
    for doc in docs:
        doc.metadata.update({
            'doc_type': metadata.get('doc_type', 'Unknown'),
            'quality_tier': metadata.get('quality_tier', 'Unknown'),
            'upload_date': datetime.datetime.now().isoformat(),
            'filename': uploaded_file.name
        })
    
    return docs, metadata

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Deck Clinic",
    page_icon="🎠",
    layout="wide"
)

# --- 2. ENHANCED CSS STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;900&family=DM+Sans:wght@400;500;700&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'DM Sans', sans-serif;
        background: linear-gradient(to bottom, #fafafa 0%, #f5f5f5 100%);
    }
    
    h1 { 
        font-family: 'Playfair Display', serif; 
        font-weight: 900;
        font-size: 3.5rem;
        color: #1a1a1a;
        letter-spacing: -2px;
        margin-bottom: 0.5rem;
    }
    
    h2, h3, h4, h5 { 
        font-family: 'Playfair Display', serif; 
        font-weight: 700;
        color: #1a1a1a;
        letter-spacing: -1px;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 3.5rem;
        font-family: 'Playfair Display', serif;
        font-weight: 900;
        background: rgba(255, 255, 255, 0.95);
        padding: 28px 20px;
        border-radius: 16px;
        border: 1px solid rgba(0, 0, 0, 0.08);
        box-shadow: 
            0 4px 6px rgba(0, 0, 0, 0.03),
            0 10px 20px rgba(0, 0, 0, 0.05);
        text-align: center;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    div[data-testid="stMetricValue"]:hover {
        transform: translateY(-2px);
        box-shadow: 
            0 8px 12px rgba(0, 0, 0, 0.05),
            0 16px 32px rgba(0, 0, 0, 0.08);
    }
    
    div[data-testid="stMetricLabel"] {
        font-family: 'DM Sans', sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 0.7rem;
        font-weight: 700;
        color: #6b7280;
        margin-bottom: 8px;
    }
    
    div.stButton > button {
        border-radius: 12px;
        border: 2px solid #1a1a1a;
        font-weight: 700;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        background-color: #1a1a1a;
        color: #ffffff;
        padding: 12px 32px;
        font-family: 'DM Sans', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.85rem;
    }
    
    div.stButton > button:hover {
        background-color: #ffffff;
        color: #1a1a1a;
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.12);
    }
    
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #ec4899 0%, #8b5cf6 100%);
        border: none;
        color: #ffffff;
    }
    
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #db2777 0%, #7c3aed 100%);
        color: #ffffff;
        box-shadow: 0 12px 24px rgba(236, 72, 153, 0.3);
    }
    
    .issue-tag {
        background-color: #fff1f2;
        color: #be123c;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.75rem;
        display: inline-block;
        margin-bottom: 10px;
        border-left: 3px solid #be123c;
        font-family: 'DM Sans', sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .fix-tag {
        background-color: #ecfdf5;
        color: #047857;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.75rem;
        display: inline-block;
        margin-bottom: 10px;
        border-left: 3px solid #047857;
        font-family: 'DM Sans', sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .logic-footer {
        font-size: 0.9rem;
        color: #4b5563;
        background: linear-gradient(to right, #fafafa 0%, #ffffff 100%);
        padding: 16px;
        border-radius: 12px;
        margin-top: 12px;
        border: 1px solid rgba(0, 0, 0, 0.06);
        font-family: 'DM Sans', sans-serif;
        line-height: 1.6;
    }
    
    .score-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 8px;
        font-family: 'DM Sans', sans-serif;
    }
    
    div[data-testid="stMarkdownContainer"] > div > div.stAlert {
        border-radius: 12px;
        border-left-width: 4px;
        font-family: 'DM Sans', sans-serif;
    }
    
    div[data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
        padding: 20px;
        border: 2px dashed rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    div[data-testid="stFileUploader"]:hover {
        border-color: rgba(0, 0, 0, 0.3);
        background: rgba(255, 255, 255, 1);
    }
    
    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 12px;
        border: 1px solid rgba(0, 0, 0, 0.06);
        margin-bottom: 16px;
    }
    
    button[data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.85rem;
    }
    
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 1px solid rgba(0, 0, 0, 0.08);
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

if not os.path.exists("user_uploads"):
    os.makedirs("user_uploads")

# --- 4. CORE ENGINE ---
@st.cache_resource
def get_embedding_model():
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

embeddings = get_embedding_model()
PERSIST_DIRECTORY = "deck_memory_db"

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("🎛️ SETTINGS")
    doc_type = st.selectbox("DIAGNOSTIC PROTOCOL", ["Strategy Deck (McKinsey/Amazon)", "Product Spec (Technical)"])
    st.divider()
    
    st.caption("📂 KNOWLEDGE BASE")
    uploaded_file = st.file_uploader("Upload 'Gold Standard' PDF", type="pdf")
    if uploaded_file and st.button("TRAIN SYSTEM"):
        with st.spinner("Indexing with metadata..."):
            docs, metadata = enhanced_document_indexing(uploaded_file)
            vector_db = Chroma.from_documents(docs, embeddings, persist_directory=PERSIST_DIRECTORY)
            try: vector_db.persist()
            except: pass
            st.success(f"✅ System Index Updated: {len(docs)} chunks")
            st.json(metadata)

    st.divider()
    
    # ADMIN PANEL
    with st.expander("🔐 ADMIN PANEL (MASTER VIEW)"):
        admin_pass = st.text_input("Enter Admin Key", type="password")
        if admin_pass == "gemini2025": 
            st.success("ACCESS GRANTED")
            has_logs = os.path.exists("clinic_logs.csv")
            has_feedback = os.path.exists("feedback_logs.csv")
            
            if has_logs:
                try:
                    df_system = pd.read_csv("clinic_logs.csv", on_bad_lines='skip')
                    df_system.columns = df_system.columns.str.strip()
                    if has_feedback:
                        df_feedback = pd.read_csv("feedback_logs.csv", on_bad_lines='skip')
                        df_feedback.columns = df_feedback.columns.str.strip()
                        df_master = pd.merge(df_system, df_feedback[['Session ID', 'Rating', 'Comment']], on='Session ID', how='left')
                        st.dataframe(df_master)
                    else:
                        st.dataframe(df_system)
                except Exception as e:
                    st.error(f"⚠️ Error reading logs: {e}")
            else:
                st.info("📭 Database is clean.")
            
            st.divider()
            if st.button("🔴 HARD RESET (Clear All Data)", type="primary"):
                if os.path.exists("clinic_logs.csv"): os.remove("clinic_logs.csv")
                if os.path.exists("feedback_logs.csv"): os.remove("feedback_logs.csv")
                for f in os.listdir("user_uploads"):
                    os.remove(os.path.join("user_uploads", f))
                st.rerun()
    
    # RAG DEBUG PANEL
    with st.expander("🔍 RAG DEBUG PANEL"):
        if 'rag_queries' in st.session_state:
            st.markdown("### Generated Search Queries")
            for i, query in enumerate(st.session_state.rag_queries, 1):
                st.code(f"{i}. {query}")

# --- 6. MAIN INTERFACE ---
st.title("🎠 DECK Clinic")
st.caption(f"Built by Olivia Li | PROTOCOL: {doc_type} | CORE: gemini-flash-latest | Advanced RAG + Chain-of-Thought") 

col1, col2 = st.columns([2, 3]) 

with col1:
    st.markdown("### UPLOAD DRAFT DECK")
    target_pdf = st.file_uploader("Upload Draft PDF", type="pdf", key="target")
    
    if not target_pdf:
        st.info("👆 Upload your deck to begin analysis")
    
    analyze_btn = st.button("RUN DIAGNOSTIC", type="primary", use_container_width=True)

# Reset Session if NEW file uploaded
if target_pdf and 'last_uploaded' not in st.session_state:
    st.session_state.last_uploaded = target_pdf.name
    st.session_state.analysis_data = None
    st.session_state.images = None
    st.session_state.session_id = str(uuid.uuid4())[:8] 
elif target_pdf and st.session_state.get('last_uploaded') != target_pdf.name:
    st.session_state.last_uploaded = target_pdf.name
    st.session_state.analysis_data = None
    st.session_state.images = None
    st.session_state.session_id = str(uuid.uuid4())[:8]

# Main Logic Flow
if (target_pdf and analyze_btn) or (target_pdf and st.session_state.get('analysis_data')):
    
    # PHASE A: GENERATION
    if not st.session_state.get('analysis_data'):
        
        session_id = st.session_state.session_id
        safe_filename = f"{session_id}_{target_pdf.name}"
        save_path = os.path.join("user_uploads", safe_filename)
        
        with open(save_path, "wb") as f:
            f.write(target_pdf.getbuffer())
        
        # Convert PDF to Images
        with st.spinner("Processing Vision (Converting Slides)..."):
            try:
                images = convert_from_path(save_path)
                st.session_state.images = images
            except Exception as e:
                st.warning(f"""
                ⚠️ **Vision processing unavailable** (Poppler not installed)
                
                Continuing with text-only analysis. For full multimodal analysis, install Poppler:
                - Mac: `brew install poppler`
                - Linux: `apt-get install poppler-utils`
                """)
                st.session_state.images = None
        
        # Text Extraction
        loader = PyPDFLoader(save_path)
        draft_docs = loader.load()
        draft_text = ""
        for i, doc in enumerate(draft_docs):
            draft_text += f"\n\n--- [PAGE {i+1}] ---\n{doc.page_content}"

        # ADVANCED RAG Retrieval
        with st.spinner("Retrieving Context (Advanced RAG)..."):
            try:
                vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
                knowledge_context, search_queries, retrieved_chunks = advanced_rag_pipeline(
                    draft_text=draft_text,
                    doc_type=doc_type,
                    vector_db=vector_db
                )
                st.session_state.rag_queries = search_queries
            except Exception as e:
                st.warning(f"Advanced RAG failed, using fallback: {e}")
                knowledge_context = "Standard Top Tech Company Protocols"

        # Enhanced CoT Prompt
        base_instruction = ""
        if "Strategy" in doc_type:
            base_instruction = "ROLE: Head of Product Manager in Tech Company. FRAMEWORK: Amazon Clarity, McKinsey Structure.BLUF, Extreme Brevity."
        elif "Product" in doc_type:
            base_instruction = "ROLE: Head of Technical PM. FRAMEWORK: Feasibility checks, Spec strictness."

        prompt = f"""
{base_instruction}

### GOLD STANDARD CONTEXT:
{knowledge_context}

### DRAFT TEXT:
{draft_text[:500000]} 

### VISUAL INPUT:
I have provided images of the slides. Please analyze them alongside the text.

### ANALYSIS FRAMEWORK (THINK STEP-BY-STEP):

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 1: DEEP REASONING (Think Out Loud)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Before scoring, analyze each dimension thoroughly:

**LOGIC REASONING:**
Step 1a: Identify the main claim/hypothesis in the deck
Step 1b: List the evidence provided (data, examples, charts)
Step 1c: Evaluate the logical chain - Does the evidence PROVE the claim? Are there gaps?
Step 1d: Check for circular reasoning or correlation≠causation errors
Step 1e: Write your reasoning (3-4 sentences explaining your logic assessment)

**CLARITY REASONING:**
Step 2a: Count abstract/vague terms (synergies, optimization, leverage, etc.)
Step 2b: Assess cognitive load - Word count per slide, concepts per slide, jargon
Step 2c: Test the "10-second rule" - Can an exec understand the slide quickly?
Step 2d: Check for visual clutter
Step 2e: Write your reasoning (3-4 sentences explaining clarity assessment)

**IMPACT REASONING:**
Step 3a: Identify the "so what?" moment
Step 3b: Check for specificity - Vague vs concrete goals
Step 3c: Assess emotional resonance - Is there urgency? Clear problem?
Step 3d: Write your reasoning (3-4 sentences explaining impact assessment)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 2: SCORING (Based on Your Reasoning Above)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NOW, based on the reasoning you just wrote:
- Logic Score: 0-100 (Is the argument sound?)
- Clarity Score: 0-100 (Can a busy exec understand it?)
- Impact Score: 0-100 (Will this change minds?)

**CRITICAL:** Your scores MUST align with your reasoning.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PHASE 3: EXTRACTION & RECONSTRUCTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3. **EXTRACTION:** Extract the current headlines to identify the existing narrative.
4. **Headline & Narrative Audit:**
   - Critique the current headlines: Do they tell a story if read in isolation?
   - Suggest a **"Revised Headline Flow"**: Rewritten headlines that guide logically from problem to solution.
5. **CONTENT RIGOR:** Scan body paragraphs, bullets, charts for vague claims.

### EXAMPLES OF GOOD CRITIQUES (FEW-SHOT):

input_text: "The KSP is enable Shopee buyers to see an AI generated summary of available promotions and encourage them to buy. In this deck, we will discuss the logic of the input of promotion summary first, then show the front end demo and share the examples of different generated example in words."
critique: "1. Grammar: 'is enable' is broken. 2. Weak Metrics: 'encourage to buy' is vague; use 'conversion'. 3. Illogical Flow: The proposed agenda jumps from 'Input Logic' to 'Frontend Demo' before validating the output quality."
rewrite: "Objective: Increase Shopee conversion rates by displaying AI-generated promotion summaries. This deck follows a three-part structure: 1. Core Logic (How inputs drive summaries), 2. Output Validation (Reviewing generated text examples), and 3. User Experience (Frontend demo)."

input_text: "We will leverage synergies to optimize the flywheel."
critique: "Jargon overload. Low clarity. No distinct meaning."
rewrite: "We will migrate the Promotion admin to CMT to significantly improve efficiency."

input_text: "Slide Title: Strong User Growth. Body: We saw significant uplift in daily active users across various regions due to better performance."
critique: "Vague Body Content. The headline is fine, but the bullet point lacks evidence. 'Significant uplift' needs a % or absolute number."
rewrite: "Body: DAU increased by 15% (20k users) in SEA and LATAM, driven by a 200ms reduction in app load time."

### JSON STRUCTURE (ENHANCED WITH REASONING):
{{
    "chain_of_thought": {{
        "logic_reasoning": "<string: Your 3-4 sentence reasoning from PHASE 1>",
        "clarity_reasoning": "<string: Your 3-4 sentence reasoning from PHASE 1>",
        "impact_reasoning": "<string: Your 3-4 sentence reasoning from PHASE 1>"
    }},
    "scores": {{
        "Logic": <int 0-100>,
        "Clarity": <int 0-100>,
        "Impact": <int 0-100>
    }},
    "executive_summary": "<string: Brutal one-sentence summary based on the chain_of_thought>",
    "narrative_check": {{
         "original_headlines": [ "<string: Extracted Headline 1>", "<string: Extracted Headline 2>" ],
         "critique": "<string: Critique of the current storytelling flow>",
         "revised_headlines": [ "<string: Improved Headline 1>", "<string: Improved Headline 2>" ]
    }},
   "section_deep_dive": [
        {{
            "page_number": "<int: The page number>",
            "target_section": "<string: Quote the specific BULLET POINT or SENTENCE>",
            "issue": "<string: Specific critique>",
            "improved_version": "<string: Rewrite>",
            "why": "<string: Why this is better>"
        }}
    ]
}}
"""

        # Generation
        with st.spinner("Processing Logic & Vision..."):
            try:
                model = genai.GenerativeModel('gemini-flash-latest')
                
                content_list = [prompt]
                if st.session_state.images:
                    content_list.extend(st.session_state.images)
                
                response = model.generate_content(
                    content_list, 
                    generation_config={"response_mime_type": "application/json"}
                )
                
                try:
                    st.session_state.analysis_data = json.loads(response.text)
                except json.JSONDecodeError:
                    cleaned = response.text.replace("```json", "").replace("```", "").strip()
                    st.session_state.analysis_data = json.loads(cleaned)
                
                log_interaction(
                    session_id=session_id,
                    filename=safe_filename,
                    doc_type=doc_type,
                    scores=st.session_state.analysis_data.get('scores', {}),
                    exec_summary=st.session_state.analysis_data.get('executive_summary', 'N/A')
                )
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.stop()

    # PHASE B: RENDERING
    data = st.session_state.analysis_data
    
    with col2:
        st.markdown(f"### SCORECARD (ID: `{st.session_state.session_id}`)")
        s1, s2, s3 = st.columns(3)
        
        logic_score = data.get('scores', {}).get('Logic', 0)
        clarity_score = data.get('scores', {}).get('Clarity', 0)
        impact_score = data.get('scores', {}).get('Impact', 0)
        
        with s1:
            st.metric("LOGIC", f"{logic_score}")
            tier, color, advice = get_score_context(logic_score)
            st.markdown(f'<div class="score-badge" style="background-color: {color}; color: white;">{tier}</div>', unsafe_allow_html=True)
            
        with s2:
            st.metric("CLARITY", f"{clarity_score}")
            tier, color, advice = get_score_context(clarity_score)
            st.markdown(f'<div class="score-badge" style="background-color: {color}; color: white;">{tier}</div>', unsafe_allow_html=True)
            
        with s3:
            st.metric("IMPACT", f"{impact_score}")
            tier, color, advice = get_score_context(impact_score)
            st.markdown(f'<div class="score-badge" style="background-color: {color}; color: white;">{tier}</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # CHAIN OF THOUGHT DISPLAY
    with st.expander("🧠 SEE AI'S REASONING (Chain-of-Thought)", expanded=False):
        st.markdown("### How the AI Arrived at These Scores")
        
        cot_data = data.get('chain_of_thought', {})
        
        st.markdown("#### 🔍 Logic Assessment")
        st.info(cot_data.get('logic_reasoning', 'No reasoning provided'))
        st.metric("Logic Score", logic_score)
        
        st.divider()
        
        st.markdown("#### 📖 Clarity Assessment")
        st.info(cot_data.get('clarity_reasoning', 'No reasoning provided'))
        st.metric("Clarity Score", clarity_score)
        
        st.divider()
        
        st.markdown("#### 💥 Impact Assessment")
        st.info(cot_data.get('impact_reasoning', 'No reasoning provided'))
        st.metric("Impact Score", impact_score)
        
        st.divider()
        
        avg_score = (logic_score + clarity_score + impact_score) / 3
        if avg_score >= 75:
            st.success("✅ High confidence in analysis - reasoning is consistent")
        elif avg_score >= 50:
            st.warning("⚠️ Moderate confidence - some areas need attention")
        else:
            st.error("🚨 Low scores detected - significant improvements recommended")
    
    st.divider()
    
    # FEEDBACK LOOP
    with st.container():
        st.markdown("#### 💬 Was this helpful?")
        fb_col1, fb_col2 = st.columns([3, 1])
        with fb_col1:
            user_comment = st.text_input("Feedback (Optional)", placeholder="E.g. The logic check was too harsh...")
        with fb_col2:
            st.write("") 
            st.write("")
            b1, b2 = st.columns(2)
            if b1.button("👍"):
                log_feedback(st.session_state.session_id, "Positive", user_comment, doc_type)
                st.toast("Feedback Saved!", icon="👍")
            if b2.button("👎"):
                log_feedback(st.session_state.session_id, "Negative", user_comment, doc_type)
                st.toast("Feedback Saved!", icon="📉")

    st.divider()
    st.info(f"**EXECUTIVE SUMMARY:** {data.get('executive_summary', 'No summary generated.')}")
    
    # TABS
    tab1, tab2 = st.tabs(["STORY FLOW", "🔬 DEEP DIVE & REWRITES"])
    
    with tab1:
        st.markdown("#### The Narrative Check (Pyramid Principle)")
        nav_data = data.get('narrative_check', {})
        st.markdown(f"> *{nav_data.get('critique', 'No critique available.')}*")
        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption("🔴 ORIGINAL FLOW")
            for line in nav_data.get('original_headlines', []): st.text(f"• {line}")
        with col_b:
            st.caption("🟢 OPTIMIZED FLOW")
            for line in nav_data.get('revised_headlines', []): st.markdown(f"**• {line}**")
        
        if logic_score < 75: st.error("⚠️ NARRATIVE THREAD BROKEN")
        else: st.success("✅ NARRATIVE THREAD STABLE")

    with tab2:
        st.markdown("#### 🔬 Surgical Reconstruction")
        st.caption("Specific text edits to improve Logic, Clarity, and Impact.")
        
        images = st.session_state.images
        
        for i, item in enumerate(data.get('section_deep_dive', [])):
            with st.container():
                page_num = item.get('page_number', '?')
                target = item.get('target_section', 'General Logic')
                
                header_text = f"📄 Page {page_num}: {target}"
                if len(header_text) > 60: header_text = header_text[:150]
                st.markdown(f"##### {header_text}")
                
                if images:
                    try:
                        p_idx = int(page_num) - 1
                        if 0 <= p_idx < len(images):
                            with st.expander(f"👁️ View Slide {page_num} Snapshot"):
                                st.image(images[p_idx], use_container_width=True)
                    except:
                        pass

                c1, c2 = st.columns([1, 2], gap="large")
                with c1:
                    st.markdown('<div class="issue-tag">THE SYMPTOM (ISSUE)</div>', unsafe_allow_html=True)
                    st.markdown(f"**{item.get('issue', 'N/A')}**")
                    st.markdown(f"<div class='logic-footer'><b>💡 ROOT CAUSE:</b><br>{item.get('why', 'N/A')}</div>", unsafe_allow_html=True)
                with c2:
                    st.markdown('<div class="fix-tag">THE PRESCRIPTION (REWRITE)</div>', unsafe_allow_html=True)
                    rewrite_text = item.get('improved_version', 'N/A')
                    if len(rewrite_text) < 300: st.info(rewrite_text, icon="✍️")
                    else:
                        st.info(rewrite_text[:300] + "...", icon="✍️")
                        with st.expander("Show Full Rewrite"): st.code(rewrite_text, language="text")
                st.divider()