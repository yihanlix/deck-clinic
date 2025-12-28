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

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Deck Clinic",
    page_icon="🎠",
    layout="wide"
)

# --- 2. ENHANCED CSS STYLING ---
st.markdown("""
<style>
    /* TYPOGRAPHY: Editorial Serif + Clean Sans */
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
    
    /* SCORE CARDS: Premium Glass Effect */
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
    
    /* BUTTONS: Sophisticated Minimal */
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
    
    /* ISSUE/FIX TAGS: Editorial Style */
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
    
    /* Score Context Badge */
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
    
    /* Info Boxes */
    div[data-testid="stMarkdownContainer"] > div > div.stAlert {
        border-radius: 12px;
        border-left-width: 4px;
        font-family: 'DM Sans', sans-serif;
    }
    
    /* File Uploader */
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
    
    /* Expander */
    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 12px;
        border: 1px solid rgba(0, 0, 0, 0.06);
        margin-bottom: 16px;
    }
    
    /* Tabs */
    button[data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.85rem;
    }
    
    /* Divider */
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

# --- 6. MAIN INTERFACE ---
st.title("🎠 DECK Clinic")
st.caption(f"Built by Olivia Li | PROTOCOL: {doc_type} | CORE: gemini-flash-latest | EMBEDDING: embedding-001 | Langchain") 

col1, col2 = st.columns([2, 3]) 

with col1:
    st.markdown("### UPLOAD DRAFT DECK")
    target_pdf = st.file_uploader("Upload Draft PDF", type="pdf", key="target")
    
    if not target_pdf:
        st.info("👆 Upload your deck to begin analysis")
    
    analyze_btn = st.button("RUN DIAGNOSTIC", type="primary", use_container_width=True)

# 1. Reset Session if NEW file uploaded
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

# 2. Main Logic Flow
if (target_pdf and analyze_btn) or (target_pdf and st.session_state.get('analysis_data')):
    
    # PHASE A: GENERATION
    if not st.session_state.get('analysis_data'):
        
        # A. File Processing
        session_id = st.session_state.session_id
        safe_filename = f"{session_id}_{target_pdf.name}"
        save_path = os.path.join("user_uploads", safe_filename)
        
        with open(save_path, "wb") as f:
            f.write(target_pdf.getbuffer())
        
        # --- CONVERT PDF TO IMAGES with Better Error Handling ---
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
        
        # B. Text Extraction for RAG
        loader = PyPDFLoader(save_path)
        draft_docs = loader.load()
        draft_text = ""
        for i, doc in enumerate(draft_docs):
            draft_text += f"\n\n--- [PAGE {i+1}] ---\n{doc.page_content}"

        # C. RAG Retrieval
        with st.spinner("Retrieving Context..."):
            try:
                vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
                results = vector_db.similarity_search(draft_text, k=3)
                knowledge_context = "\n".join([doc.page_content for doc in results])
            except:
                knowledge_context = "Standard Top Tech Company Protocols"

        # D. Prompt Construction (YOUR ORIGINAL PROMPT)
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
        
        ### INSTRUCTIONS:
        1. **STEP 1 (HIDDEN BRAINSTORM):** Read the text AND look at the images. Look for logical gaps and visual clutter. Ask yourself: "Does the problem prove the solution?" "Is the data specific?" "Does the chart support the headline?"
        2. **STEP 2 (SCORING):** Only assign scores AFTER you have written the critique.
        3. **STEP 3 (EXTRACTION):** Extract the current headlines to identify the existing narrative.
        4. **STEP 4 (Headline & Narrative Audit):**
           - Critique the current headlines: Do they tell a story if read in isolation? Are they descriptive or generic?
           - Suggest a **"Revised Headline Flow"**: A list of rewritten headlines that guide the reader logically from the problem to the solution.
        5. **STEP 5 (CONTENT RIGOR):** Scan the **body paragraphs, bullet points, and charts** for vague claims (e.g., "significant growth", "optimized synergies").
       
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
                 "original_headlines": [ "<string: Extracted Headline 1>", "<string: Extracted Headline 2>" ],
                 "critique": "<string: Critique of the current storytelling flow>",
                 "revised_headlines": [ "<string: Improved Headline 1>", "<string: Improved Headline 2>" ]
            }},
           "section_deep_dive": [
                {{
                    "page_number": "<int: The page number extracted from the [PAGE X] marker>",
                    "target_section": "<string: Quote the specific BULLET POINT or SENTENCE (not the headline)>",
                    "issue": "<string: Specific critique of the evidence/data OR Visual issue>",
                    "improved_version": "<string: Rewrite the bullet point to be data-driven and specific>",
                    "why": "<string: Why this is better>"
                }}
            ]
        }}
        """

        # E. Generation (Multimodal with Better Error Handling)
        with st.spinner("Processing Logic & Vision..."):
            try:
                model = genai.GenerativeModel('gemini-flash-latest')
                
                # Build content list based on whether images are available
                content_list = [prompt]
                if st.session_state.images:
                    content_list.extend(st.session_state.images)
                
                response = model.generate_content(
                    content_list, 
                    generation_config={"response_mime_type": "application/json"}
                )
                
                # Parse JSON with error handling
                try:
                    st.session_state.analysis_data = json.loads(response.text)
                except json.JSONDecodeError:
                    # Try cleaning up markdown fences
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
        
        # Enhanced Score Display with Context
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
    
    # TABS RENDER
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
        
        if data.get('scores', {}).get('Logic', 0) < 75: st.error("⚠️ NARRATIVE THREAD BROKEN")
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
                
                # Show slide image if available
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