# This file is used to ingest the pre-processed dataset into a vector database
# Execute this file only once. Otherwise, the vector database will be overwritten.
# Import necessary libraries

import os
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Define the class for vector database
class VectorDatabase:
    # Initialize the constructor
    def __init__(self, data_path, db_faiss_path):
        self.data_path = data_path
        self.db_faiss_path = db_faiss_path

    # Create the vector database function
    def create_vector_db(self):
        # Create the directory if it doesn't exist. Ensure parent directory exists
        os.makedirs(os.path.dirname(self.db_faiss_path), exist_ok=True)

        # Load the data
        print(f"Loading data from {self.data_path}...")
        loader = CSVLoader(file_path=self.data_path, encoding="utf-8")
        documents = loader.load()

        # Split the data
        print("Text splitting...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
        texts = text_splitter.split_documents(documents)

        # Embedding the data using HuggingFace
        print("Embedding documents...")
        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'}
        )

        # Create the vector database
        print("Creating vector database...")
        try:
            db = FAISS.from_documents(texts, embeddings)
            print(db)
            db.save_local(self.db_faiss_path)
            print("Vector database created successfully")
        except Exception as e:
            print(f"Error creating vector database: {e}")


# Define the main function
if __name__ == "__main__":
    vector_db = VectorDatabase(
        data_path="data/pre-processed_dataset.csv",
        db_faiss_path="vectorstore/db_faiss",
    )
    vector_db.create_vector_db()
