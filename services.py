import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

model = SentenceTransformer("BAAI/bge-base-en-v1.5", device='cpu')

full_text = ''
with pdfplumber.open("2023050195.pdf") as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        full_text += text + '\n'


def chunk_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

chunks = chunk_text(full_text, chunk_size=1000, overlap=100)
print("Total chunks:", len(chunks))



embeddings = model.encode(chunks, convert_to_tensor=False, show_progress_bar=True)

print("Embeddings created:", len(embeddings))
print("Embedding dimension:", len(embeddings[0]))



embedding_dim = len(embeddings[0])
embedding_matrix = np.array(embeddings).astype('float32')
index = faiss.IndexFlatL2(embedding_dim) 
index.add(embedding_matrix)
print("FAISS index has", index.ntotal, "vectors")
faiss.write_index(index, "pdf_embeddings.index")
with open("pdf_chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Embeddings and chunks saved locally ✅")

