from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

# Define the paths to the files
BASE_PATH = Path("/home/kaisinyru/Downloads/chatbot_TVTS/AI_lord/data")

def create_and_save_faiss_index(documents: List, name: str, embeddings, output_dir: Path):
    vector = FAISS.from_documents(documents, embeddings)
    vector.save_local(str(output_dir / f"faiss_index_{name}"))

def process_directory(base_path: Path, output_dir: Path):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=125, length_function=len)
    embeddings = OpenAIEmbeddings()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get total number of files
    total_files = sum(1 for _ in base_path.glob("**/*") if _.is_file())

    # Iterate over all files in the directory with a progress bar
    for file_path in tqdm(base_path.glob("**/*"), total=total_files, desc="Processing files"):
        if file_path.is_file():
            # Create a TextLoader for each file
            loader = TextLoader(str(file_path))
            
            # Load the document
            documents = loader.load()
            
            # Process each document
            for doc in documents:
                name = file_path.stem
                
                split_documents = text_splitter.split_documents([doc])
                create_and_save_faiss_index(split_documents, name, embeddings, output_dir)

if __name__ == "__main__":
    output_directory = Path("vector_database")
    process_directory(BASE_PATH, output_directory)