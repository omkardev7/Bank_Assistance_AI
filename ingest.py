# ingest.py
import json
import os
import re
from langchain_exa import ExaSearchResults
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import config

logger = config.get_logger(__name__)

def clean_text(text):
    """Enhanced text cleaning function"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep essential punctuation
    text = re.sub(r'[^\w\s\.\,\-\%\(\)\:\/]', '', text)
    
    # Remove URLs except domain names
    text = re.sub(r'https?://[^\s]+', lambda m: m.group(0).split('/')[2] if '/' in m.group(0) else m.group(0), text)
    
    # Normalize percentage format
    text = re.sub(r'(\d+)\s*%', r'\1%', text)
    
    # Remove repeated phrases
    lines = text.split('.')
    seen = set()
    unique_lines = []
    for line in lines:
        normalized = line.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_lines.append(line.strip())
    
    return '. '.join(unique_lines)

def append_to_file(filepath, content):
    """Helper to append text to a file with error handling"""
    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(content + "\n\n")
    except Exception as e:
        logger.error(f"Error writing to {filepath}: {e}")
        raise

def parse_exa_result(raw_result_str):
    """Enhanced parsing with better error handling"""
    clean_content = ""
    formatted_content = ""
    
    # First, try to parse as JSON
    try:
        data = json.loads(raw_result_str)
        
        if "results" in data and isinstance(data["results"], list):
            for idx, item in enumerate(data["results"]):
                title = item.get("title", "No Title")
                url = item.get("url", "No URL")
                text = item.get("text", "")
                
                # Clean the text content
                cleaned_text = clean_text(text)
                
                if cleaned_text:
                    segment = f"Source: {title}\nURL: {url}\nContent: {cleaned_text}"
                    clean_content += segment + "\n\n"
                    
                    formatted_content += f"=== DOCUMENT {idx + 1} ===\n"
                    formatted_content += f"TITLE: {title}\n"
                    formatted_content += f"URL: {url}\n"
                    formatted_content += f"CONTENT:\n{cleaned_text}\n"
                    formatted_content += "=" * 50 + "\n"
        else:
            logger.warning("Unexpected data structure in Exa result")
            clean_content = clean_text(str(raw_result_str))
            formatted_content = f"=== UNSTRUCTURED DATA ===\n{clean_content}\n"
            
    except json.JSONDecodeError:
        # Not JSON - handle as structured text from Exa
        logger.info("Result is not JSON format, parsing as structured text")
        
        # Parse the structured text format that Exa returns
        raw_text = str(raw_result_str)
        
        # Extract structured information using regex patterns
        import re
        
        # Split by document boundaries
        doc_pattern = r'Title: (.*?)(?:\n|$).*?URL: (.*?)(?:\n|$).*?Text: (.*?)(?=Title:|$)'
        matches = re.findall(doc_pattern, raw_text, re.DOTALL)
        
        if matches:
            for idx, (title, url, text) in enumerate(matches):
                cleaned_text = clean_text(text.strip())
                
                if cleaned_text:
                    segment = f"Source: {title.strip()}\nURL: {url.strip()}\nContent: {cleaned_text}"
                    clean_content += segment + "\n\n"
                    
                    formatted_content += f"=== DOCUMENT {idx + 1} ===\n"
                    formatted_content += f"TITLE: {title.strip()}\n"
                    formatted_content += f"URL: {url.strip()}\n"
                    formatted_content += f"CONTENT:\n{cleaned_text}\n"
                    formatted_content += "=" * 50 + "\n"
        else:
            # Fallback: treat entire result as one document
            cleaned_text = clean_text(raw_text)
            clean_content = f"Source: Exa Search Result\nContent: {cleaned_text}"
            formatted_content = f"=== RAW TEXT ===\n{cleaned_text}\n"
            
    except Exception as e:
        logger.error(f"Unexpected error parsing Exa result: {e}")
        # Last resort fallback
        clean_content = clean_text(str(raw_result_str))
        formatted_content = f"=== ERROR RECOVERY ===\n{clean_content}\n"
    
    return clean_content, formatted_content

def fetch_and_process_data():
    """Fetch data from Exa with enhanced error handling"""
    logger.info("Starting data collection and processing")
    
    # Clean up old files
    for filepath in [config.get_raw_data_path(), config.get_cleaned_data_path()]:
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Removed existing file: {filepath}")
    
    try:
        exa_tool = ExaSearchResults(
            exa_api_key=config.get_exa_api_key(),
            num_results=config.get_exa_num_results(),
            use_autoprompt=True
        )
    except Exception as e:
        logger.error(f"Failed to initialize Exa tool: {e}")
        raise
    
    queries = [
        "site:bankofmaharashtra.bank.in interest rates home loan",
        "site:bankofmaharashtra.bank.in personal loan eligibility",
        "site:bankofmaharashtra.bank.in Maha Super Flexi Housing Loan",
        "site:bankofmaharashtra.bank.in processing fee home loan",
        "site:bankofmaharashtra.bank.in education loan",
        "site:bankofmaharashtra.bank.in vehicle loan rates"
    ]
    
    documents = []
    successful_queries = 0
    
    for query in queries:
        logger.info(f"Processing query: {query}")
        try:
            raw_result = exa_tool.invoke(query)
            
            # Save raw data
            append_to_file(
                config.get_raw_data_path(),
                f"QUERY: {query}\nRAW_RESULT:\n{raw_result}"
            )
            
            # Parse and clean
            clean_content, formatted_content = parse_exa_result(str(raw_result))
            
            # Save cleaned data
            append_to_file(
                config.get_cleaned_data_path(),
                f"QUERY: {query}\n{formatted_content}"
            )
            
            if clean_content.strip():
                doc = Document(
                    page_content=clean_content,
                    metadata={
                        "query": query,
                        "source": "Bank of Maharashtra"
                    }
                )
                documents.append(doc)
                successful_queries += 1
                logger.info(f"Successfully processed: {query}")
            else:
                logger.warning(f"No content extracted from: {query}")
                
        except Exception as e:
            logger.error(f"Failed to process query '{query}': {e}")
            continue
    
    logger.info(f"Completed: {successful_queries}/{len(queries)} queries successful")
    return documents

def create_vector_db(documents):
    """Create FAISS vector database with error handling"""
    if not documents:
        logger.error("No documents to ingest")
        raise ValueError("No documents available for vector database creation")
    
    try:
        logger.info("Starting text chunking")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.get_chunk_size(),
            chunk_overlap=config.get_chunk_overlap(),
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        
        logger.info("Creating embeddings")
        embeddings = HuggingFaceEmbeddings(
            model_name=config.get_embedding_model()
        )
        
        logger.info("Building FAISS index")
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        db_path = config.get_vector_db_path()
        vector_store.save_local(db_path)
        logger.info(f"Vector database saved to: {db_path}")
        
    except Exception as e:
        logger.error(f"Error creating vector database: {e}")
        raise

if __name__ == "__main__":
    try:
        docs = fetch_and_process_data()
        create_vector_db(docs)
        logger.info("Data ingestion completed successfully")
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        raise