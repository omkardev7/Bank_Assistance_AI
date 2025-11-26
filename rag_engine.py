#rag_engine.py
import os
import json
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import config

logger = config.get_logger(__name__)

# Global cache for models
_retriever = None
_llm = None
_prompt = None

def get_enhanced_prompt():
    """Enhanced prompt with better instructions"""
    template = """You are an expert Loan Assistant for Bank of Maharashtra with deep knowledge of banking products.

CONTEXT INFORMATION:
{context}

CONVERSATION HISTORY:
{history}

CURRENT QUESTION: {question}

INSTRUCTIONS:
1. Provide accurate, specific information based ONLY on the context provided
2. Always cite specific interest rates, fees, and eligibility criteria when available
3. If information is not in the context, clearly state: "Based on available documentation, I don't have that information. Please contact the bank directly."
4. Never invent or assume numbers, rates, or terms
5. Format your response clearly with bullet points for multiple items
6. If the question relates to previous conversation, reference it naturally
7. Be professional, helpful, and concise

ANSWER:"""
    
    return ChatPromptTemplate.from_template(template)

def initialize_system():
    """Initialize RAG system components"""
    global _retriever, _llm, _prompt
    
    if _retriever is not None:
        logger.info("System already initialized")
        return
    
    try:
        logger.info("Initializing RAG system")
        
        # Load embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name=config.get_embedding_model()
        )
        
        # Load vector store
        db_path = config.get_vector_db_path()
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Vector database not found at {db_path}. Run ingest.py first.")
        
        vector_store = FAISS.load_local(
            db_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        _retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.get_retrieval_k()}
        )
        logger.info("Vector store loaded successfully")
        
        # Initialize LLM
        _llm = ChatGoogleGenerativeAI(
            model=config.get_llm_model(),
            temperature=config.get_llm_temperature(),
            google_api_key=config.get_google_api_key()
        )
        logger.info("LLM initialized successfully")
        
        # Setup prompt
        _prompt = get_enhanced_prompt()
        logger.info("RAG system initialization complete")
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise

def format_conversation_history(history):
    """Format conversation history for context"""
    if not history:
        return "No previous conversation"
    
    formatted = []
    for entry in history[-3:]:  # Last 3 exchanges
        formatted.append(f"User: {entry['question']}")
        formatted.append(f"Assistant: {entry['answer'][:200]}...")  # Truncate long answers
    
    return "\n".join(formatted)

def save_conversation(session_id, question, answer, context):
    """Save conversation to file"""
    try:
        conv_dir = config.get_conversation_history_path()
        os.makedirs(conv_dir, exist_ok=True)
        
        filepath = os.path.join(conv_dir, f"{session_id}.json")
        
        conversation = []
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                conversation = json.load(f)
        
        conversation.append({
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "context_sources": len(context)
        })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        logger.error(f"Failed to save conversation: {e}")

def load_conversation(session_id):
    """Load conversation history"""
    try:
        filepath = os.path.join(config.get_conversation_history_path(), f"{session_id}.json")
        
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    except Exception as e:
        logger.error(f"Failed to load conversation: {e}")
        return []

def validate_question(question):
    """Validate and clean the question"""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
    
    # Remove excessive whitespace
    question = " ".join(question.split())
    
    # Check minimum length
    if len(question) < 3:
        raise ValueError("Question is too short")
    
    return question

def process_query(question, session_id="default"):
    """Process query with conversation history support"""
    try:
        # Ensure system is initialized
        if _retriever is None:
            initialize_system()
        
        # Validate question
        question = validate_question(question)
        logger.info(f"Processing query: {question[:100]}")
        
        # Load conversation history
        history = load_conversation(session_id)
        history_text = format_conversation_history(history)
        
        # Retrieve relevant documents
        docs = _retriever.invoke(question)
        logger.info(f"Retrieved {len(docs)} relevant documents")
        
        if not docs:
            logger.warning("No relevant documents found")
            return {
                "answer": "I couldn't find relevant information in the bank's documentation. Please contact Bank of Maharashtra directly for assistance.",
                "context_used": [],
                "sources": []
            }
        
        # Format context
        context_text = "\n\n---\n\n".join([
            f"Source {i+1}:\n{doc.page_content}"
            for i, doc in enumerate(docs)
        ])
        
                # 🌟 NEW: Print full prompt with actual injected values
        final_prompt = _prompt.format(
            context=context_text,
            history=history_text,
            question=question
        )
        # logger.info("\n================ FINAL PROMPT SENT TO LLM ================\n")
        # logger.info(final_prompt)
        # logger.info("\n===========================================================\n")

        
        # Generate answer
        chain = _prompt | _llm | StrOutputParser()
        
        answer = chain.invoke({
            "context": context_text,
            "history": history_text,
            "question": question
        })
        
        logger.info("Answer generated successfully")
        
        # Prepare context snippets
        context_snippets = [
            doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            for doc in docs
        ]
        
        # Extract source metadata
        sources = [
            doc.metadata.get("source", "Bank of Maharashtra")
            for doc in docs
        ]
        
        # Save conversation
        save_conversation(session_id, question, answer, context_snippets)
        
        return {
            "answer": answer,
            "context_used": context_snippets,
            "sources": list(set(sources))  # Unique sources
        }
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise

def clear_conversation(session_id):
    """Clear conversation history for a session"""
    try:
        filepath = os.path.join(config.get_conversation_history_path(), f"{session_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Cleared conversation for session: {session_id}")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to clear conversation: {e}")
        return False