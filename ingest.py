import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader

#load our environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

#Define embedding model
EMBEDDING_MODEL = "gemini-embedding-2"

chroma_client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name = "company_documents"
)

def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages):
        text = page.extract_text()

        if text:
            pages.append({
                "page": page_number + 1,
                "text": text
            })

    return pages

def split_into_chunks(text, chunk_size=500, overlap=50):
    words = text.split()

    chunks = []

    start=0

    while start < len(words):
        end = start + chunk_size

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

def generate_embedding(text):
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text
    )

    return response.embeddings[0].values

def ingest_document(pdf_path):
    print(f"Processing: {pdf_path}")

    pages = extract_pdf_text(pdf_path)

    total_chunks = 0

    for page in pages:
        chunks = split_into_chunks(page["text"])

        for chunk_number, chunk in enumerate(chunks):
            embedding = generate_embedding(chunk)

            document_id = (
                f"{Path(pdf_path).stem}_"
                f"{page['page']}_"
                f"{chunk_number}"
            )

            collection.upsert(
                ids=[document_id],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{
                    "source": Path(pdf_path).name,
                    "page": page["page"]
                }]
            )

            total_chunks += 1

    print(f"Successfully stored {total_chunks} chunks.")

if __name__ == "__main__":
    documents_folder = Path("documents")

    pdf_files = list(documents_folder.glob("*.pdf"))

    if not pdf_files:
        print("No PDF documents found.")
    else:
        for pdf_file in pdf_files:
            ingest_document(pdf_file)

    print("Document ingestion completed.")

    client.close()