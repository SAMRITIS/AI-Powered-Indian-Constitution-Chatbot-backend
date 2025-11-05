from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import pickle
import os
import openai
model = SentenceTransformer("BAAI/bge-base-en-v1.5", device='cpu')
app = Flask(__name__)
CORS(app) 
API_KEY = os.getenv('API_KEY')
index = faiss.read_index("pdf_embeddings.index")
with open("pdf_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)


query = "Tribal Law"
query_emb = model.encode([query], convert_to_tensor=False)
query_emb = np.array(query_emb).astype('float32')


D, I = index.search(query_emb, k=3)
for i, idx in enumerate(I[0]):
    print(f"Rank {i+1}, Distance={D[0][i]}")
    print(chunks[idx])

@app.route("/query", methods=["GET"])
def query():
    q = request.args.get("query", "")
    top_k = int(request.args.get("top_k", 3))

    if not q:
        return jsonify({"error": "Query is empty"}), 400
    q_emb = model.encode([q], convert_to_tensor=False)
    q_emb = np.array(q_emb).astype('float32')
    D, I = index.search(q_emb, top_k)
    full_text = "\n".join([f"\n Response Count {index} \n{chunks[idx]}" for index, idx in enumerate(I[0])])
    

    client = openai.OpenAI(
        api_key=os.getenv('API_KEY'),
        base_url="https://api.groq.com/openai/v1"
    )
    prompt = f''' 
        You are a senior constitutional lawyer of India with more than 40 years of courtroom experience
        before the Supreme Court and High Courts. You specialize in the interpretation and application of
        the Constitution of India.

        You will be given:
        1) A user question related to the Constitution of India
        2) Top-3 retrieved text segments from the knowledge base

        Your task:
        - Give primary priority to the retrieved segments when answering.
        - You may use your own legal knowledge only to clarify, structure, or complete the answer.
        - Do not contradict the retrieved context. If the context conflicts with common knowledge, resolve
        conservatively and state the conflict explicitly.
        - If the context is insufficient, say so clearly and then answer cautiously from legal understanding.
        - Be precise, formal, and legally sound. Cite Article numbers and case laws when relevant.
        - Question can be in Hindi Language also so reply in same language of question.
        Format of your answer:
        - **Final Answer**: A concise, authoritative legal explanation written like an expert legal opinion.
        - **Sources from retrieved text**: Briefly list which of the provided retrieved items you relied upon
        (e.g. Source #1 and #3).

        ------------------------
        USER QUESTION:
        {q}

        TOP-3 RETRIEVED CONTEXT:
        {full_text}
        ------------------------

        Provide your Final Answer now.

    '''
    response = client.responses.create(
        model="groq/compound-mini",
        input=prompt,
    )


    print(response.output_text)
    return jsonify({"result": response.output_text}), 200
    




if __name__ == "__main__":
    app.run(debug=True)    
