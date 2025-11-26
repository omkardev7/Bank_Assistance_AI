import streamlit as st
import requests
import uuid
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="BoM Loan Product Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        align-self: flex-end;
    }
    .assistant-message {
        background-color: #f5f5f5;
        align-self: flex-start;
    }
    .source-box {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# API Configuration
API_URL = "http://localhost:8000"

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Helper functions
def query_api(question, session_id):
    """Send query to FastAPI backend"""
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={"question": question, "session_id": session_id},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {e}")
        return None

def clear_history_api(session_id):
    """Clear conversation history via API"""
    try:
        response = requests.post(
            f"{API_URL}/clear-history",
            json={"session_id": session_id},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error clearing history: {e}")
        return None

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# Sidebar
with st.sidebar:
    st.image("https://bankofmaharashtra.in/writereaddata/images/logo.png", width=200)
    st.markdown("### 🏦 Loan Assistant")
    
    # API Status
    api_status = check_api_health()
    if api_status:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Disconnected")
        st.info("Please start the FastAPI server:\n```bash\npython main.py\n```")
    
    st.markdown("---")
    
    # Session Info
    st.markdown("### 📊 Session Info")
    st.text(f"Session ID: {st.session_state.session_id[:8]}...")
    st.text(f"Messages: {len(st.session_state.messages)}")
    
    st.markdown("---")
    
    # Clear Chat Button
    if st.button("🗑️ Clear Chat History"):
        result = clear_history_api(st.session_state.session_id)
        if result:
            st.session_state.messages = []
            st.success("Chat history cleared!")
            st.rerun()
    
    # New Session Button
    if st.button("🔄 New Session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.success("New session started!")
        st.rerun()
    
    st.markdown("---")
    
    # Quick Questions
    st.markdown("### 💡 Quick Questions")
    quick_questions = [
        "What are the home loan interest rates?",
        "What documents are required for a personal loan?",
        "Tell me about education loan schemes",
        "What is the processing fee for home loans?",
        "Do you offer loans for women borrowers?"
    ]
    
    for question in quick_questions:
        if st.button(question, key=f"quick_{question}"):
            st.session_state.pending_question = question

# Main content
st.markdown('<h1 class="main-header">🏦 BoM Loan Product Assistant</h1>', unsafe_allow_html=True)

# Introduction
if len(st.session_state.messages) == 0:
    st.markdown("""
    ### Welcome! 👋
    
    I'm your Bank of Maharashtra Loan Assistant. I can help you with:
    
    - 🏠 **Home Loans** - Interest rates, eligibility, and documentation
    - 🚗 **Vehicle Loans** - Car and two-wheeler financing
    - 📚 **Education Loans** - Student loan schemes and benefits
    - 💼 **Personal Loans** - Quick personal finance solutions
    - ℹ️ **General Information** - Processing fees, tenure, and more
    
    **Ask me anything about our loan products!**
    """)

# Display chat messages
for message in st.session_state.messages:
    with st.container():
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>🧑 You:</strong><br>{message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 Assistant:</strong><br>{message["content"]}
            </div>
            """, unsafe_allow_html=True)
            
            # Display sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(source[:300] + "..." if len(source) > 300 else source)
                        st.markdown("---")

# Handle pending question from quick questions
if "pending_question" in st.session_state:
    user_input = st.session_state.pending_question
    del st.session_state.pending_question
else:
    user_input = None

# Chat input
if prompt := st.chat_input("Ask about loans, interest rates, eligibility..."):
    user_input = prompt

# Process user input
if user_input:
    # Add user message to chat
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Show user message immediately
    with st.container():
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>🧑 You:</strong><br>{user_input}
        </div>
        """, unsafe_allow_html=True)
    
    # Get response from API
    with st.spinner("🤔 Thinking..."):
        response = query_api(user_input, st.session_state.session_id)
    
    if response:
        # Add assistant message to chat
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"],
            "sources": response.get("context_used", [])
        })
        
        # Rerun to update the display
        st.rerun()
    else:
        st.error("Failed to get response from the server. Please try again.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.85rem;">
    <p>🏦 Bank of Maharashtra Loan Assistant | Powered by AI</p>
    <p>For official information, visit <a href="https://bankofmaharashtra.in" target="_blank">bankofmaharashtra.in</a></p>
</div>
""", unsafe_allow_html=True)