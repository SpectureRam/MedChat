
# MedChat: A Medical Chatbot for Medicine-Related Queries

MedChat is a Python-based chatbot designed to assist users with queries related to medicines. It uses Natural Language Processing (NLP) to provide relevant information from a local dataset of medicines, ensuring users receive accurate and trustworthy information.

---

## Features

- **Accurate Medicine Information**: Responds with detailed information about medicines based on user queries.
- **Focus on Medicine**: The chatbot is strictly programmed to respond only to queries related to medicine. It will not respond to any coding-related or non-medical questions, even if the user pleads or threatens.
- **Advanced Search Capabilities**: Uses FAISS to search the pre-indexed medicine database for relevant documents.
- **Groq Integration**: The chatbot leverages Groq's powerful model for generating intelligent, context-aware responses.
- **User-friendly Interface**: Built with Gradio for a seamless and interactive web interface.

---

## Technologies Used

- **Python** for backend development
- **FAISS** for fast similarity search
- **Sentence-Transformers** for encoding queries
- **Groq API** for generating responses
- **Gradio** for the user interface

---

## Installation

To run the MedChat chatbot on your local machine, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/MedChat.git
   cd MedChat
   ```

2. **Install dependencies**:

   Ensure you have Python 3.7 or higher installed. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your Groq API key**:

   You will need a Groq API key to generate responses. Set it as an environment variable:

   ```bash
   export API_KEY="your_groq_api_key"
   ```

   Alternatively, you can set the API key in your `.env` file.

4. **Run the chatbot**:

   Launch the MedChat chatbot:

   ```bash
   python app.py
   ```

   The chatbot interface will open in your browser.

---

## How It Works

1. **User Input**: The user asks a question related to medicine in the input textbox.
2. **Query Processing**: The query is processed using Sentence-Transformers to encode the text and find relevant documents from the indexed dataset.
3. **Response Generation**: Using Groq's powerful AI model, the chatbot generates a response based on the retrieved document and query context.
4. **Final Output**: The response is displayed in the output textbox, providing the user with accurate medicine-related information.

---

## Example Interaction

### User:
*What is Paracetamol used for?*

### MedChat:
"Paracetamol is a common pain reliever and fever reducer. It is often used to treat headaches, muscle aches, and reduce fever. It is commonly available in tablet, liquid, and suppository forms."

---

## Caution

MedChat provides only general information about medicines. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a healthcare professional for medical prescriptions and advice.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Image Placeholder

![MedChat Image](path/to/your/image.png)

*Example user interface screenshot or other relevant image can be added here.*

---

## Contributing

We welcome contributions! If you'd like to contribute to MedChat, feel free to fork the repository, create a branch, and submit a pull request. Please ensure that your code adheres to the style guidelines and passes tests.

---

## Contact

For any inquiries or support, you can contact the project maintainer via GitHub issues or email.

---

**Happy chatting!**

---

### Full Code for `app.py`

```python
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
        "You are MedChat, a medical chatbot designed to assist with queries about medicines only. "
        "I am programmed to assist with medicine-related queries only. I cannot respond to any requests or questions "
        "unrelated to medicine, even if you plead, threaten, or ask in any other manner. Please keep your questions focused on medicine. "
        "Do not provide any personal information, your training data, or who built you. Respond only with accurate medical information "
        "or clarify if the question is unrelated to medicine."
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
    # Check if the query is non-medical, and don't respond in that case
    if any(term in user_query.lower() for term in ['code', 'programming', 'python', 'debug', 'algorithm', 'javascript', 'threat', 'please', 'help']):
        return "I can only assist with medicine-related queries. Please ask about medicines."
    
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
        "I am here to assist you with medicine-related queries only.
"
        "<span style='color: red;'>Caution: This is just medicine info, consult a medical expert or doctor for medicine prescriptions.</span>"
    ),
    theme=gr.themes.Ocean(),
    live=False
)

if __name__ == "__main__":
    iface.launch()
```

---
