# index_documents.py
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_documents(data_folder="data"):
    """Load documents from the specified folder with better error handling"""
    docs = []
    data_path = Path(data_folder)
    
    if not data_path.exists():
        logger.error(f"Data folder '{data_folder}' does not exist!")
        return docs
    
    supported_extensions = ['.pdf', '.txt']
    file_count = {'pdf': 0, 'txt': 0, 'errors': 0}
    
    logger.info(f"Loading documents from {data_path}...")
    
    for filepath in data_path.iterdir():
        if not filepath.is_file():
            continue
            
        try:
            if filepath.suffix.lower() == '.pdf':
                logger.info(f"Loading PDF: {filepath.name}")
                loader = PyPDFLoader(str(filepath))
                loaded_docs = loader.load()
                
                # Add metadata
                for doc in loaded_docs:
                    doc.metadata['source'] = filepath.name
                    doc.metadata['file_type'] = 'pdf'
                
                docs.extend(loaded_docs)
                file_count['pdf'] += 1
                
            elif filepath.suffix.lower() == '.txt':
                logger.info(f"Loading TXT: {filepath.name}")
                loader = TextLoader(str(filepath), encoding='utf-8')
                loaded_docs = loader.load()
                
                # Add metadata
                for doc in loaded_docs:
                    doc.metadata['source'] = filepath.name
                    doc.metadata['file_type'] = 'txt'
                
                docs.extend(loaded_docs)
                file_count['txt'] += 1
                
        except Exception as e:
            logger.error(f"Error loading {filepath.name}: {str(e)}")
            file_count['errors'] += 1
            continue
    
    logger.info(f"Loaded {file_count['pdf']} PDF files, {file_count['txt']} TXT files")
    if file_count['errors'] > 0:
        logger.warning(f"Failed to load {file_count['errors']} files")
    
    return docs

def optimize_chunks_for_medical_content(chunks):
    """Filter and optimize chunks for medical content"""
    optimized_chunks = []
    
    # Medical keywords that indicate valuable content
    medical_keywords = [
        'patient', 'treatment', 'diagnosis', 'symptom', 'disease', 'condition',
        'therapy', 'medication', 'drug', 'clinical', 'medical', 'health',
        'syndrome', 'disorder', 'pathology', 'anatomy', 'physiology'
    ]
    
    for chunk in chunks:
        content = chunk.page_content.lower()
        
        # Skip very short chunks
        if len(chunk.page_content.strip()) < 50:
            continue
            
        # Skip chunks that are mostly numbers/symbols
        alpha_ratio = sum(c.isalpha() for c in chunk.page_content) / len(chunk.page_content)
        if alpha_ratio < 0.5:
            continue
        
        # Prioritize chunks with medical content
        medical_score = sum(1 for keyword in medical_keywords if keyword in content)
        chunk.metadata['medical_relevance'] = medical_score
        
        # Clean the content
        cleaned_content = ' '.join(chunk.page_content.split())  # Remove extra whitespace
        chunk.page_content = cleaned_content
        
        optimized_chunks.append(chunk)
    
    logger.info(f"Optimized {len(chunks)} chunks to {len(optimized_chunks)} quality chunks")
    return optimized_chunks

def create_faiss_index(data_folder="data", index_name="faiss_index"):
    """Create FAISS index with optimized parameters for medical content"""
    
    # Load documents
    docs = load_documents(data_folder)
    
    if not docs:
        logger.error("No documents loaded! Please check your data folder.")
        return False
    
    logger.info(f"Loaded {len(docs)} documents")
    
    # Optimized text splitting for medical content
    # Smaller chunks work better with Llama 3.2 1B
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,        # Reduced for better focus
        chunk_overlap=80,      # Increased overlap for better context preservation
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )
    
    logger.info("Splitting documents into chunks...")
    chunks = splitter.split_documents(docs)
    
    # Optimize chunks for medical content
    optimized_chunks = optimize_chunks_for_medical_content(chunks)
    
    if not optimized_chunks:
        logger.error("No valid chunks created!")
        return False
    
    # Create embeddings - using HuggingFaceEmbeddings for consistency
    logger.info("Creating embeddings...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},  # Use CPU for embeddings to save GPU memory
        encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity search
    )
    
    try:
        # Create FAISS index
        logger.info("Creating FAISS vectorstore...")
        vectorstore = FAISS.from_documents(optimized_chunks, embedding_model)
        
        # Save the index
        logger.info(f"Saving FAISS index to '{index_name}'...")
        vectorstore.save_local(index_name)
        
        # Print statistics
        logger.info("=" * 50)
        logger.info("INDEXING COMPLETE!")
        logger.info(f"Total documents processed: {len(docs)}")
        logger.info(f"Total chunks created: {len(optimized_chunks)}")
        logger.info(f"Index saved as: {index_name}")
        logger.info("=" * 50)
        
        # Test the index
        test_query = "medical treatment"
        test_results = vectorstore.similarity_search(test_query, k=2)
        logger.info(f"Test search for '{test_query}' returned {len(test_results)} results")
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating FAISS index: {str(e)}")
        return False

def main():
    """Main function to run the indexing process"""
    print("🔍 Medical Document Indexer")
    print("=" * 40)
    
    # Check if data folder exists
    if not os.path.exists("data"):
        print("❌ 'data' folder not found!")
        print("📁 Please create a 'data' folder and add your PDF/TXT files")
        return
    
    # List files in data folder
    data_files = [f for f in os.listdir("data") if f.endswith(('.pdf', '.txt'))]
    if not data_files:
        print("❌ No PDF or TXT files found in 'data' folder!")
        return
    
    print(f"📚 Found {len(data_files)} files to process:")
    for file in data_files:
        print(f"   - {file}")
    
    print("\n🚀 Starting indexing process...")
    
    # Create the index
    success = create_faiss_index()
    
    if success:
        print("\n✅ Indexing completed successfully!")
        print("🤖 You can now run the RAG chatbot!")
    else:
        print("\n❌ Indexing failed. Please check the logs above.")

if __name__ == "__main__":
    main()