# app.py (Flask Backend - FIXED VERSION)

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from pypdf import PdfReader
import numpy as np
import faiss 
import io
import re

# --- 1. FLASK SETUP & CONFIGURATION 


app = Flask(__name__)
CORS(app) # Allow React frontend to access this API

# Load API Key from environment variable
try:
    API_KEY = os.environ.get('GEMINI_API_KEY')
    if not API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    
    # Configure with explicit client options to use v1 API
    genai.configure(
        api_key=API_KEY,
        transport='rest',  # Use REST instead of gRPC
        client_options={'api_endpoint': 'generativelanguage.googleapis.com'}
    )
    
except Exception as e:
    print(f"FATAL ERROR: {e}")

# --- 2. GLOBAL STATE FOR RAG/CHAT ---
RAG_STATE = {
    "VECTOR_DB": None,
    "TEXT_CHUNKS": [],
    "CHAT_SESSION": None,
    "SYSTEM_INSTRUCTION": (
        "You are an expert, versatile AI assistant and Q&A system. "
        "Always generate clear, accurate, and informative answers. "
        "When document context is provided, use it as your primary source. "
        "If the question cannot be answered from the context, clearly state that and optionally provide general knowledge. "
        "Be concise but comprehensive in your responses."
    ),
    "EMBEDDING_MODEL": 'text-embedding-004',
    "PDF_FILENAME": None,
}

# --- 3. IMPROVED UTILITY FUNCTIONS ---

def get_pdf_text(file_stream):
    """Extracts text from a PDF file stream with better error handling."""
    text = ""
    try:
        reader = PdfReader(file_stream)
        print(f"PDF has {len(reader.pages)} pages")
        
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
            else:
                print(f"Warning: Page {i+1} has no extractable text")
        
        # Clean up the text
        text = clean_text(text)
        print(f"Extracted {len(text)} characters from PDF")
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

def clean_text(text):
    """Cleans extracted PDF text."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s\.,;:!?\'"-]', '', text)
    return text.strip()

def smart_chunk_text(text, chunk_size=500, overlap=100):
    """
    Intelligently chunks text by sentences while respecting chunk_size.
    Better than splitting by paragraphs which can be too large or too small.
    """
    # Split into sentences (simple approach)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence exceeds chunk_size, save current chunk
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap (last few words)
            words = current_chunk.split()
            overlap_text = ' '.join(words[-overlap//5:]) if len(words) > overlap//5 else ""
            current_chunk = overlap_text + " " + sentence
        else:
            current_chunk += " " + sentence
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    print(f"Created {len(chunks)} chunks from text")
    return chunks

def create_embeddings(text):
    """Creates embeddings with improved chunking and error handling."""
    # Use smart chunking instead of paragraph splitting
    text_chunks = smart_chunk_text(text, chunk_size=600, overlap=100)
    
    if not text_chunks:
        print("No text chunks created")
        return None, []
    
    print(f"Processing {len(text_chunks)} chunks for embedding...")
    embeddings = []
    
    # Process in smaller batches to avoid API limits
    batch_size = 20
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        try:
            result = genai.embed_content(
                model=RAG_STATE["EMBEDDING_MODEL"],
                content=batch,
                task_type="retrieval_document"  # lowercase as per API docs
            )
            embeddings.extend(result['embedding'])
            print(f"Embedded batch {i//batch_size + 1}/{(len(text_chunks)-1)//batch_size + 1}")
        except Exception as e:
            print(f"Error embedding batch {i//batch_size + 1}: {e}")
            # Continue with other batches instead of failing completely
            continue
    
    if not embeddings:
        print("No embeddings created")
        return None, []
    
    # Create FAISS index
    embeddings_array = np.array(embeddings, dtype='float32')
    dimension = embeddings_array.shape[1]
    vector_db = faiss.IndexFlatL2(dimension)
    vector_db.add(embeddings_array)
    
    print(f"FAISS index created with {vector_db.ntotal} vectors")
    return vector_db, text_chunks

def retrieve_relevant_context(user_prompt, vector_db, text_chunks, top_k=5):
    """Retrieves the most relevant chunks for the user's query."""
    try:
        # Create query embedding
        query_embedding_result = genai.embed_content(
            model=RAG_STATE["EMBEDDING_MODEL"],
            content=user_prompt,
            task_type="retrieval_query"  # lowercase
        )
        query_vector = np.array([query_embedding_result['embedding']], dtype='float32')
        
        # Search for similar chunks
        k = min(top_k, len(text_chunks))  # Don't search for more chunks than we have
        distances, indices = vector_db.search(query_vector, k)
        
        # Get the chunks and their relevance scores
        retrieved_chunks = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(text_chunks):  # Safety check
                retrieved_chunks.append({
                    'text': text_chunks[idx],
                    'distance': float(distance)
                })
        
        print(f"Retrieved {len(retrieved_chunks)} relevant chunks")
        return retrieved_chunks
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return []

def generate_rag_response(user_prompt, retrieved_chunks, chat_session):
    """Generates response using retrieved context."""
    if not retrieved_chunks:
        # No relevant context found
        prompt = f"""
        I don't have specific information from the document to answer this question.
        
        Question: {user_prompt}
        
        Please provide a response based on general knowledge, or indicate if you need the document context.
        """
    else:
        # Build context from retrieved chunks
        context_parts = [chunk['text'] for chunk in retrieved_chunks]
        context_string = "\n\n---\n\n".join(context_parts)
        
        prompt = f"""
        Based on the following context from the document, please answer the question.
        If the context doesn't contain enough information, say so clearly.
        
        DOCUMENT CONTEXT:
        {context_string}
        
        QUESTION: {user_prompt}
        
        Please provide a clear, accurate answer based on the context above.
        """
    
    response = chat_session.send_message(prompt)
    return response.text

# --- 4. API ENDPOINTS ---

@app.route('/new_chat', methods=['POST'])
def new_chat():
    """Initializes a new chat session without clearing the document."""
    try:
        # Use models that are actually available in your account
        # From your list: gemini-2.5-flash, gemini-2.0-flash, gemini-flash-latest
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",  # This IS in your available models
            system_instruction=RAG_STATE["SYSTEM_INSTRUCTION"],
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
        )
        print(f"Successfully initialized model: gemini-2.5-flash")
        
        # Start a new chat session
        RAG_STATE["CHAT_SESSION"] = model.start_chat(history=[])
        
        # Note: Document context is preserved unless explicitly cleared
        
        return jsonify({
            "message": "New chat session started.",
            "document_loaded": RAG_STATE["VECTOR_DB"] is not None,
            "filename": RAG_STATE["PDF_FILENAME"]
        }), 200
    except Exception as e:
        print(f"Error starting chat: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_pdf():
    """Handles PDF file upload and creates the RAG vector store."""
    if 'pdf_file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    pdf_file = request.files['pdf_file']
    if pdf_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        # Read PDF content from in-memory stream
        pdf_stream = io.BytesIO(pdf_file.read())
        pdf_text = get_pdf_text(pdf_stream)
        
        if not pdf_text or len(pdf_text) < 100:
            return jsonify({
                "error": "Failed to extract meaningful text from PDF. The file might be scanned/image-based or corrupted."
            }), 400

        # Create Embeddings
        vector_db, text_chunks = create_embeddings(pdf_text)
        
        if not vector_db or not text_chunks:
            return jsonify({
                "error": "Failed to create embeddings. Please try again."
            }), 500
        
        # Update Global State
        RAG_STATE["VECTOR_DB"] = vector_db
        RAG_STATE["TEXT_CHUNKS"] = text_chunks
        RAG_STATE["PDF_FILENAME"] = pdf_file.filename
        
        status_msg = f"PDF '{pdf_file.filename}' successfully processed."
        print(status_msg)
        
        return jsonify({
            "message": status_msg,
            "chunks": len(text_chunks),
            "filename": pdf_file.filename
        }), 200
    
    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({"error": f"Upload failed: {str(e)}"}), 500

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """Handles incoming chat messages and generates response using RAG or general knowledge."""
    if not RAG_STATE["CHAT_SESSION"]:
        return jsonify({"error": "Chat session not initialized. Call /new_chat first."}), 400
    
    data = request.get_json()
    user_prompt = data.get('message', '').strip()
    
    if not user_prompt:
        return jsonify({"error": "Message field is empty"}), 400

    try:
        if RAG_STATE["VECTOR_DB"] and RAG_STATE["TEXT_CHUNKS"]:
            # Use RAG with the document context
            print(f"Processing query with RAG: {user_prompt[:50]}...")
            
            retrieved_chunks = retrieve_relevant_context(
                user_prompt, 
                RAG_STATE["VECTOR_DB"], 
                RAG_STATE["TEXT_CHUNKS"],
                top_k=5
            )
            
            response_text = generate_rag_response(
                user_prompt,
                retrieved_chunks,
                RAG_STATE["CHAT_SESSION"]
            )
        else:
            # Use standard chat (no document loaded)
            print(f"Processing query without RAG: {user_prompt[:50]}...")
            response = RAG_STATE["CHAT_SESSION"].send_message(user_prompt)
            response_text = response.text

        return jsonify({
            "response": response_text,
            "used_rag": RAG_STATE["VECTOR_DB"] is not None
        }), 200

    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/clear_document', methods=['POST'])
def clear_document():
    """Clears the loaded document but keeps the chat session."""
    RAG_STATE["VECTOR_DB"] = None
    RAG_STATE["TEXT_CHUNKS"] = []
    RAG_STATE["PDF_FILENAME"] = None
    return jsonify({"message": "Document cleared. Chat session maintained."}), 200

@app.route('/status', methods=['GET'])
def get_status():
    """Returns current system status."""
    return jsonify({
        "chat_initialized": RAG_STATE["CHAT_SESSION"] is not None,
        "document_loaded": RAG_STATE["VECTOR_DB"] is not None,
        "filename": RAG_STATE["PDF_FILENAME"],
        "chunks": len(RAG_STATE["TEXT_CHUNKS"])
    }), 200


if __name__ == '__main__':
    # Initialize a default chat session on server start
    print("Initializing Flask app...")
    
    # Check package version
    import google.generativeai as genai_check
    print(f"google-generativeai version: {genai_check.__version__}")
    
    # List available models
    print("\n=== Available Gemini Models ===")
    try:
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                print(f"✓ {model.name}")
    except Exception as e:
        print(f"Could not list models: {e}")
    print("================================\n")
    
    with app.app_context():
        try:
            new_chat()
            print("Default chat session initialized")
        except Exception as e:
            print(f"Failed to initialize chat: {e}")
    
    # Run the Flask app
    print("Starting Flask server on port 5000...")
    app.run(debug=True, port=5000, host='127.0.0.1')