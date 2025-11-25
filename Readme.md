
# 🏦 Loan Product Assistant – RAG-Based Chatbot

An AI-powered Retrieval-Augmented Generation (RAG) system that provides accurate, context-grounded answers about BoM's loan products.
This project integrates **FastAPI**, **LangChain**, **FAISS**, **Gemini**, **HuggingFace Embeddings**, **Exa Search**, and a **Streamlit UI**.

---



## 🚀 Project Structure

```
.
├── app.py                 # Streamlit UI
├── main.py                # FastAPI backend
├── rag_engine.py          # RAG pipeline logic
├── ingest.py              # Exa search → cleaning → FAISS builder
├── config.py              # Environment variables & configs
├── faiss_index_bom/       # Vector DB directory (generated)
├── conversations/         # Chat history storage (generated)
├── loan_data_raw.txt      # Raw Exa data dump
├── loan_data_cleaned.txt  # Cleaned dataset
└── README.md
```

---

## ⚙️ Setup 

### Prerequisites

- Python 3.10 or higher
- pip package manager
- API Keys:
  - Google Gemini API Key
  - EXA Search API Key

### 1️⃣ **Clone Repository**

```bash
git clone https://github.com/omkardev7/Bank_Assistance_AI.git
cd Bank_Assistance_AI
```

### 2️⃣ **Create Virtual Environment**

```bash
python -m venv venv

source venv/bin/activate  # Mac/Linux

venv\Scripts\activate     # Windows
```

### 3️⃣ **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 4️⃣ **Add Environment Variables**

Create a `.env` file:

```
GOOGLE_API_KEY=your_google_api_key
EXA_API_KEY=your_exa_api_key
```

## 📖 Usage Instructions

### 1️⃣ **Run Data Ingestion**

Fetch loan-related data and build FAISS index:

```bash
python ingest.py
```

### 2️⃣ **Start FastAPI Backend**

```bash
python main.py
```

Server will run at:
👉 **[http://localhost:8000](http://localhost:8000)**

### 3️⃣ **Start Streamlit Frontend**

```bash
streamlit run app.py
```

UI available at:
👉 **[http://localhost:8501](http://localhost:8501)**

---

## 📡 API Endpoints

### **1. GET /**

Health response
→ `{ status: "ok" }`

### **2. POST /query**

**Request:**

```json
{
  "question": "What are home loan interest rates?",
  "session_id": "abc123"
}
```

**Response:**

```json
{
  "answer": "...",
  "context_used": ["..."],
  "sources": ["Bank of Maharashtra"],
  "session_id": "abc123"
}
```

### **3. POST /clear-history**

Clears stored chat history for the given session.

---

## 📊 Technologies Used

| Component            | Technology          |
| -------------------- | ------------------- |
| Backend API          | FastAPI             |
| Frontend             | Streamlit           |
| LLM                  | Google Gemini  |
| Embeddings           | HuggingFace all-MiniLM-L6  |
| Vector DB            | FAISS               |
| Search Provider      | Exa Search          |
| Prompting            | LangChain           |
| Conversation Storage | JSON files          |

---

## 📁 Data Ingestion Flow

Exa Search → Raw Data → Cleaning → Deduplication → Chunking → Embeddings → FAISS Index

This ensures:
✔ Clean dataset
✔ High retrieval accuracy
✔ Proper chunk sizes for RAG

---
## 🧠 RAG Working Flow

1. Receive user question
2. Validate + clean input
3. Load last 3 exchanges from conversation history
4. Retrieve similar documents using FAISS
5. Build a combined enhanced prompt
6. Gemini generates a grounded response
7. Return answer + retrieved context to frontend
8. Save conversation for future continuity

---
## 📌 Features

### 🔍 **Retrieval-Augmented Generation (RAG)**

* Uses **FAISS vector database** for document retrieval
* Embeddings generated using **sentence-transformers/all-MiniLM-L6-v2**
* Ensures answers are **factual**, **contextual**, and **non-hallucinated**

### 🤖 **LLM Integration**

* Powered by **Google Gemini 2.5 Flash**
* Follows a strict, enhanced prompt for accurate banking responses

### 🧠 **Conversation Memory**

* Maintains session-based chat history
* Saves and loads past conversation context
* Per-session `.json` files stored locally

### 🌐 **FastAPI Backend**

* Endpoints for query processing, health check, and clearing chat memory
* Error handling and validation built-in

### 💻 **Streamlit Frontend**

* Clean, interactive chat UI
* Quick questions, session controls, and source explanations
* Displays retrieved context snippets

### 🌐 **Automated Data Ingestion**

* Fetches official Bank of Maharashtra loan data using **Exa Search**
* Cleans, parses, and structures documents
* Builds FAISS vector store for RAG

---
