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

def main():
    """Main function to test Chroma retrieval."""
    print("Setting up OpenAI API key...")
    setup_openai_api_key()
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # Connect to existing vector store
    vector_store = Chroma(
        collection_name="knowledge_base",
        embedding_function=embeddings,
        persist_directory="./chroma_persist_dir",
    )
    
    print(f"\nVector store collection: {vector_store._collection.name}")
    print(f"Number of documents in collection: {vector_store._collection.count()}")
        
    query = "2023 total revenue and quarterly revenue"
    sources = ["2023_annual_report.md", "2023_business_review_meeting_notes.md"]
    
    # Perform similarity search

    for source in sources:

        results = vector_store.similarity_search(
            query,
            k=3,
            filter={"source": source}
        )

    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}:")
        print("-" * 40)
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print("Content:")
        print(doc.page_content)
        print("-" * 40)
            

if __name__ == "__main__":
    main()
