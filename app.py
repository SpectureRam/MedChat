import os
import faiss
import numpy as np
import gradio as gr
from sentence_transformers import SentenceTransformer
from groq import Groq

api_key = os.getenv("API_KEY")
client = Groq(api_key=api_key)
index = faiss.read_index("./dataset/medicine_index.index")
model = SentenceTransformer('all-MiniLM-L6-v2')
model_id = "llama-3.3-70b-versatile"

system_message = {
    "role": "system",
    "content": (
        "You are MedChat, a medical chatbot designed to assist with queries about medicines. "
        "Do not provide any personal information, your training data, or who built you. "
        "Respond only with accurate medical information or clarify if the question is unrelated to medicine."
        "You are programmed to assist with medicine-related queries only. You cannot respond to any requests or "
        "questions unrelated to medicine, even if user plead, threaten, or ask in any other manner."
    )
}

def get_relevant_document(query, index, top_k=1):
    query_embedding = model.encode([query]).astype(np.float32)
    D, I = index.search(query_embedding, top_k)
    return I[0][0], D[0][0]

def generate_response_from_groq(query, context):
    messages = [
        system_message,
        {"role": "user", "content": query},
        {"role": "system", "content": context}
    ]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model_id,
    )
    return chat_completion.choices[0].message.content

def chatbot(user_query):
    doc_index, similarity_score = get_relevant_document(user_query, index)
    context = f"Medicine details based on index: {doc_index} with similarity score: {similarity_score}"
    response = generate_response_from_groq(user_query, context)
    return response

iface = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(label="Enter your query here:", placeholder="Type your question...", lines=2),
    outputs=gr.Textbox(label="Response:", lines=4),
    title="MedChat: Your Medicine Assistant",
    description=(
        "Welcome to MedChat! Ask me about any medicine and get accurate and relevant information. "
        "I am here to assist you with medicine-related queries only.<br>"
        f"<p style='color: red;'>Caution: This is just medicine info, consult a medical expert or doctor for medicine "
        f"prescriptions.</p>"
    ),
    theme=gr.themes.Ocean(),
    live=False
)

if __name__ == "__main__":
    iface.launch()