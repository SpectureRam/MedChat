
# MedChat: A Medical Chatbot for Medicine-Related Queries

MedChat is a Python-based chatbot designed to assist users with queries related to medicines. It uses Natural Language Processing (NLP) to provide relevant information from a local dataset of medicines, ensuring users receive accurate and trustworthy information.

---

## Application Image

![MedChat Application Screenshot](https://github.com/user-attachments/assets/2770b1e7-ba5d-42b5-968f-f655414adafe)

---
## Features

- **Accurate Medicine Information**: Responds with detailed information about medicines based on user queries.
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

## Contributing

We welcome contributions! If you'd like to contribute to MedChat, feel free to fork the repository, create a branch, and submit a pull request. Please ensure that your code adheres to the style guidelines and passes tests.

---

## Contact

For any inquiries or support, you can contact the project maintainer via GitHub issues or email.

---
