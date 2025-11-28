import streamlit as st
import os
import base64
from math_chat import MathRAG 

# ================= CONFIGURATION =================
NOTES_FOLDER = "./my_notes"
DB_PATH = "./math_notes_db"

st.set_page_config(
    page_title="Math RAG Assistant",
    page_icon="∫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CUSTOM CSS STYLING =================
def local_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        h1 { color: #4F46E5; font-weight: 700; }
        
        /* Chat bubbles */
        .stChatMessage {
            background-color: transparent;
            border-radius: 15px;
            padding: 1rem;
        }
        .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
            background-color: #f3f4f6;
        }
        @media (prefers-color-scheme: dark) {
            .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
                background-color: #1f2937; 
            }
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# ================= INITIALIZE BACKEND =================
@st.cache_resource
def get_chatbot():
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("⚠️ OPENAI_API_KEY is missing. Please set it in your environment.")
        return None
    return MathRAG(NOTES_FOLDER, DB_PATH)

bot = get_chatbot()

# ================= SIDEBAR =================
with st.sidebar:
    # REPLACED BROKEN IMAGE WITH EMOJI HEADER
    st.markdown("## 🧮 Math Notes")
    st.caption("AI-Powered Tutor")
    st.divider()
    
    # 1. File Statistics
    if os.path.exists(NOTES_FOLDER):
        num_files = len([f for f in os.listdir(NOTES_FOLDER) if f.endswith('.pdf')])
    else:
        num_files = 0
    st.metric(label="Indexed PDFs", value=num_files)
    
    st.divider()

    # 2. Controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Scan", type="primary", use_container_width=True):
            if bot:
                with st.spinner("Indexing..."):
                    bot.load_and_index_pdfs()
                    st.rerun()
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    st.caption(f"📁 Source: `{NOTES_FOLDER}`")

# ================= CHAT LOGIC =================
st.title("Math Notes Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- EMPTY STATE (Welcome Screen) ---
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div style='text-align: center; padding: 30px 0;'>
        <h3>👋 Welcome!</h3>
        <p style='color: gray;'>I can read your PDF notes and solve math problems using them.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Suggestion Chips
    c1, c2, c3 = st.columns(3)
    if c1.button("Summarize the main theorems"):
        st.session_state.messages.append({"role": "user", "content": "Summarize the main theorems in my notes."})
        st.rerun()
        
    if c2.button("Find a definition for..."):
        st.session_state.messages.append({"role": "user", "content": "Find a definition for..."})
        st.rerun()
        
    if c3.button("Create a practice problem"):
        st.session_state.messages.append({"role": "user", "content": "Create a practice problem based on the notes."})
        st.rerun()

# --- DISPLAY HISTORY ---
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑‍🎓" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])
        
        if "images" in message and message["images"]:
            with st.expander(f"📚 View {len(message['images'])} Source Pages"):
                cols = st.columns(min(3, len(message['images'])))
                for idx, img in enumerate(message["images"]):
                    col = cols[idx % 3] 
                    with col:
                        st.image(base64.b64decode(img), use_container_width=True)
                        st.caption(message["captions"][idx])

# --- CAPTURE NEW INPUT ---
if prompt := st.chat_input("Ask a math question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun() # Force a rerun to trigger the "Generate Answer" block below

# --- GENERATE ANSWER (The Fix) ---
# Check if the last message is from the user. If so, generate an answer.
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    
    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        
        with st.spinner("🧠 Reasoning with Visual Cortex..."):
            try:
                # Get last user message
                user_query = st.session_state.messages[-1]["content"]
                
                # Call Backend
                answer_text, images, captions = bot.query(user_query)
                
                # Render Text
                message_placeholder.markdown(answer_text)
                
                # Render Images (if any)
                if images:
                    with st.expander(f"📚 View {len(images)} Source Pages"):
                        cols = st.columns(min(3, len(images)))
                        for idx, img in enumerate(images):
                            col = cols[idx % 3]
                            with col:
                                st.image(base64.b64decode(img), use_container_width=True)
                                st.caption(captions[idx])

                # Save to History
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer_text,
                    "images": images,
                    "captions": captions
                })
                
            except Exception as e:
                st.error(f"An error occurred: {e}")