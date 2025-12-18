import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os
import shutil

# --- 1. SETUP & CONFIG ---
st.set_page_config(page_title="Deck Clinic: Memory Edition", page_icon="🧠", layout="wide")

# Try to get the key from Streamlit's Secret Vault
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except:
    # If running locally without secrets file, warn the user
    st.error("No API Key found in secrets!")

os.environ["GOOGLE_API_KEY"] = api_key
genai.configure(api_key=api_key)
model = genai.GenerativeModel('models/gemini-flash-latest')

# Initialize Embeddings (The tool that turns text into math)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Define where to save the "Brain" on your computer
PERSIST_DIRECTORY = "./deck_memory_db"

# --- 2. THE MEMORY FUNCTIONS (RAG) ---
def save_to_brain(uploaded_file):
    """Reads a PDF and saves it to the Vector Database"""
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    
    # 1. Chunking: We split the text so the AI can find specific parts
    # (Simple character split for MVP)
    chunk_size = 1000
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    # 2. Turn into "Documents" for LangChain
    docs = [Document(page_content=chunk) for chunk in chunks]
    
    # 3. Save to ChromaDB
    # If DB exists, append; if not, create.
    db = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        persist_directory=PERSIST_DIRECTORY
    )
    return len(chunks)

def clear_brain():
    """Deletes the memory to start fresh"""
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)

# --- 3. UI LAYOUT ---
st.title("🧠 Deck Clinic: With Corporate Memory")
st.markdown("**Compare your new draft against your company's 'Gold Standard' history.**")

# --- SIDEBAR: KNOWLEDGE BASE ---
with st.sidebar:
    st.header("📚 Knowledge Base")
    st.info("Upload your BEST past proposals here to train the AI.")
    
    training_file = st.file_uploader("Upload 'Gold Standard' PDF", type="pdf", key="train")
    
    if training_file and st.button("🧠 Train AI on this Deck"):
        with st.spinner("Embedding knowledge..."):
            num_chunks = save_to_brain(training_file)
            st.success(f"Learned! Stored {num_chunks} knowledge chunks.")
    
    if st.button("🗑️ Reset Memory"):
        clear_brain()
        st.warning("Memory wiped.")

# --- MAIN: THE REVIEW AREA ---
st.header("🚀 Review New Draft")
target_file = st.file_uploader("Upload New Proposal Draft", type="pdf", key="target")

if target_file and st.button("Analyze with Context"):
    with st.spinner("Consulting the Knowledge Base..."):
        # 1. Read the New Draft
        reader = PdfReader(target_file)
        draft_text = ""
        for page in reader.pages:
            draft_text += page.extract_text()

        # 2. RETRIEVAL: Ask the DB for relevant "Gold Standard" examples
        # We ask: "What does a good proposal look like?" based on the draft's content
        try:
            db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
            # Find 3 most similar chunks from the "Good Deck"
            results = db.similarity_search(draft_text[:2000], k=3)
            knowledge_context = "\n\n".join([doc.page_content for doc in results])
            
            st.success("✅ Retrieved relevant style guides from History.")
            with st.expander("See what the AI remembered from the Gold Deck"):
                st.write(knowledge_context)
            
            # 3. GENERATION: The Comparison Prompt
            prompt = f"""
            ROLE: You are a Senior Editor enforcing company consistency.
            
            CONTEXT (THE GOLD STANDARD):
            The following text is from a past SUCCESSFUL proposal (The "Gold Standard"):
            {knowledge_context}
            
            TASK:
            Review the "NEW DRAFT" below. 
            Compare it to the style/tone/structure of the "Gold Standard".
            
            NEW DRAFT:
            {draft_text}
            
            OUTPUT:
            1. **Style Match Score (0-100%):** How close is this to our winning style?
            2. **Gap Analysis:** What is the New Draft doing differently than the Gold Standard? (e.g., Is it too wordy? Is the data format different?)
            3. **Rewrite:** Rewrite the Introduction of the New Draft to perfectly mimic the Gold Standard's voice.
            """
            
            response = model.generate_content(prompt)
            st.markdown(response.text)
            
        except Exception as e:
            st.error("Memory Error: Did you upload a Gold Standard deck first? (DB not found)")