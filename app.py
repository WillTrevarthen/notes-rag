import streamlit as st
import os
import base64
# Import your class from your other file
from math_chat import MathRAG 

# ================= CONFIG =================
NOTES_FOLDER = "./my_notes"
DB_PATH = "./math_notes_db"
# Ensure API KEY is set in your environment or set it here
# os.environ["OPENAI_API_KEY"] = "sk-..." 

st.set_page_config(page_title="Math Notes AI", layout="wide")

# ================= INITIALIZE BACKEND =================
@st.cache_resource
def get_chatbot():
    """
    We cache this function so we don't reload the database 
    every time you click a button.
    """
    # Check if API key exists
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("⚠️ OPENAI_API_KEY is missing. Please set it in your environment.")
        return None
        
    return MathRAG(NOTES_FOLDER, DB_PATH)

bot = get_chatbot()

# ================= SIDEBAR =================
with st.sidebar:
    st.header("📚 Your Notes")
    
    if st.button("🔄 Scan & Index New PDFs"):
        if bot:
            with st.spinner("Reading PDFs... this might take a moment..."):
                # Capturing printed output from the backend is hard, 
                # so we just run the method and show a success message.
                bot.load_and_index_pdfs() 
                st.success("Indexing Complete!")
    
    st.markdown("---")
    st.write("Files are stored in:", NOTES_FOLDER)

# ================= CHAT INTERFACE =================
st.title("🧮 Math RAG Chatbot")
st.markdown("Ask questions about your notes. I will display **LaTeX** math and source images.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If images were saved with this message, show them
        if "images" in message and message["images"]:
            with st.expander("View Source Pages"):
                for idx, img in enumerate(message["images"]):
                    st.caption(message["captions"][idx])
                    st.image(base64.b64decode(img))

# Handle new user input
if prompt := st.chat_input("How do I solve..."):
    if not bot:
        st.error("Chatbot failed to initialize.")
        st.stop()

    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate Answer
    with st.chat_message("assistant"):
        with st.spinner("Analyzing notes..."):
            # Call the backend
            answer_text, images, captions = bot.query(prompt)
            
            # Display Answer (Streamlit renders the LaTeX $$ automatically)
            st.markdown(answer_text)
            
            # Display Source Images
            if images:
                with st.expander("View Source Pages Used"):
                    for idx, img in enumerate(images):
                        st.caption(captions[idx])
                        st.image(base64.b64decode(img))
    
    # 3. Save to History
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer_text,
        "images": images,
        "captions": captions
    })