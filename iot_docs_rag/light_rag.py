import os
import asyncio
from io import BytesIO
import fitz as pymupdf  # PyMuPDF
import pymupdf4llm
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
import dotenv
import httpx

# Load environment variables from .env file
dotenv.load_dotenv()

WORKING_DIR = "./iot-docs"
PDF_FILE_PATH = "/Users/sdkappasrl/Desktop/UserFolder/git/cole_medin_agentic_rag/iot_docs_rag/documents/Catalogo Axess_2025_ENG v18.0.pdf.pdf"  # Update with your actual PDF path

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        The extracted text in markdown format
    """
    try:
        doc = pymupdf.open(pdf_path)
        return pymupdf4llm.to_markdown(doc)
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {e}")


def extract_text_from_pdf_bytes(data: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF.

    Args:
        data: PDF content as bytes

    Returns:
        The extracted text in markdown format
    """
    try:
        pdf_file = BytesIO(data)
        doc = pymupdf.open(stream=pdf_file)
        return pymupdf4llm.to_markdown(doc)
    except Exception as e:
        raise Exception(f"Error extracting text from PDF bytes: {e}")


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def main():
    # Initialize RAG instance and insert PDF content
    rag = asyncio.run(initialize_rag())

    # Extract text from PDF file
    pdf_text = extract_text_from_pdf(PDF_FILE_PATH)

    # Insert the extracted text into the RAG system
    rag.insert(pdf_text)


if __name__ == "__main__":
    main()
