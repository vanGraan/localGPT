# check_dependencies.py
required_packages = [
    "langchain",
    "langchain_community",
    "pdfminer.six",
    "InstructorEmbedding",
]

for package in required_packages:
    try:
        __import__(package)
        print(f"{package} is installed.")
    except ImportError:
        print(f"{package} is NOT installed. You can install it with: pip install {package}")


# check_dependencies.py
required_packages = [
    "langchain",
    "langchain_community",
    "pdfminer.six",
    "InstructorEmbedding",
    "sentence-transformers",
]

for package in required_packages:
    try:
        __import__(package)
        print(f"{package} is installed.")
    except ImportError:
        print(f"{package} is NOT installed. You can install it with: pip install {package}")

from langchain_community.document_loaders import (
    CSVLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredExcelLoader,
    Docx2txtLoader,
    UnstructuredFileLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
)
from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
    HuggingFaceInstructEmbeddings,
    HuggingFaceBgeEmbeddings,
)
from langchain_community.vectorstores import Chroma
from langchain_community.schema import Document
