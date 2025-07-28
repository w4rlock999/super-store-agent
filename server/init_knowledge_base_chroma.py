import getpass
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv()


def setup_openai_api_key():
    """Set up OpenAI API key from environment or user input."""
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


def load_documents_from_directory(knowledge_base_dir: str) -> list[Document]:
    """Load all documents from the knowledge base directory."""
    docs = []
    
    if not os.path.exists(knowledge_base_dir):
        raise FileNotFoundError(f"Knowledge base directory not found: {knowledge_base_dir}")
    
    for filename in os.listdir(knowledge_base_dir):
        file_path = os.path.join(knowledge_base_dir, filename)
        if os.path.isfile(file_path) and filename.endswith(('.md', '.txt')):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    docs.append(Document(page_content=text, metadata={"source": filename}))
                print(f"Loaded document: {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return docs


def split_documents(docs: list[Document], chunk_size: int = 1000, chunk_overlap: int = 100) -> list[Document]:
    """Split documents into smaller chunks for better embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(docs)


def initialize_vector_store(split_docs: list[Document], collection_name: str = "knowledge_base", 
                          persist_directory: str = "./chroma_persist_dir") -> Chroma:
    """Initialize and populate the Chroma vector store."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    
    # Add documents to the vector store
    vector_store.add_documents(split_docs)
    print(f"Added {len(split_docs)} document chunks to vector store")
    
    return vector_store


def init_knowledge_base(knowledge_base_dir: str = "./knowledge_base_documents",
                       collection_name: str = "knowledge_base",
                       persist_directory: str = "./chroma_persist_dir",
                       chunk_size: int = 1000,
                       chunk_overlap: int = 110) -> Chroma:
    """
    Initialize the knowledge base by loading documents, splitting them, and creating a vector store.
    
    Args:
        knowledge_base_dir: Directory containing the knowledge base documents
        collection_name: Name for the Chroma collection
        persist_directory: Directory to persist the vector database
        chunk_size: Size of document chunks for splitting
        chunk_overlap: Overlap between chunks
        
    Returns:
        Initialized Chroma vector store
    """
    print("Setting up OpenAI API key...")
    setup_openai_api_key()
    
    print(f"Loading documents from {knowledge_base_dir}...")
    docs = load_documents_from_directory(knowledge_base_dir)
    print(f"Loaded {len(docs)} documents")
    
    print("Splitting documents into chunks...")
    split_docs = split_documents(docs, chunk_size, chunk_overlap)
    print(f"Created {len(split_docs)} document chunks")
    
    print(split_docs[25])

    # print("Initializing vector store...")
    vector_store = initialize_vector_store(split_docs, collection_name, persist_directory)
    
    # print("Knowledge base initialization complete!")
    return vector_store


def main():
    """Main function to initialize the knowledge base."""
    try:
        vector_store = init_knowledge_base()
        print(f"Vector store created successfully with collection: {vector_store._collection.name}")
    except Exception as e:
        print(f"Error initializing knowledge base: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())