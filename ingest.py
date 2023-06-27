"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle

from langchain.document_loaders import ReadTheDocsLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.document_loaders import PyPDFLoader
#from langchain.document_loaders import PyMuPDFLoader


import os

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_BASE"] = "https://x.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = ""

def ingest_docs():
    """Get documents from web pages."""

    file_paths = []
    raw_documents = []
    #for root, directories, files in os.walk("/mnt/c/Users/u44850/Downloads/extracted"):
    for root, directories, files in os.walk("/mnt/c/Users/u44850/Downloads/meag"):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)


            loader = PyPDFLoader(filepath)
            pages = loader.load()

            raw_documents.extend(pages)
    embeddings = OpenAIEmbeddings(chunk_size=1)
    vectorstore = FAISS.from_documents(raw_documents, embeddings)

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == "__main__":
    ingest_docs()
