**Medical Chatbot using Retrieval-Augmented Generation**
**Overview**

The Medical Chatbot is a conversational assistant designed to provide health-related information by referencing trusted medical resources. Instead of generating answers solely from a language model, the chatbot uses a Retrieval-Augmented Generation (RAG) pipeline, which grounds responses in actual text from medical literature. This ensures more factual and reliable answers.

The knowledge base was created from the Gale Encyclopedia of Medicine (Vol. 1). Since the source PDF was scanned and image-based, Optical Character Recognition (OCR) was used to extract clean, searchable text.

**Features**

-> Answers medical questions based on reference text.

-> Uses a RAG pipeline to minimize hallucinations.

-> Stores embeddings in Pinecone for efficient semantic search.

-> Flask-based web application for user interaction.

-> Supports local inference of models through Ollama.

-> Modular design allows adding more documents to the knowledge base.

**System Architecture**

                ┌─────────────────────────┐
                │      User (Flask UI)     │
                └─────────────┬───────────┘
                              │
                              ▼
                ┌─────────────────────────┐
                │   Query Embedding Layer  │
                │ (LangChain + Embeddings) │
                └─────────────┬───────────┘
                              │
                              ▼
                ┌─────────────────────────┐
                │   Pinecone Vector Store  │
                │ (retrieves top matches)  │
                └─────────────┬───────────┘
                              │
                              ▼
                ┌─────────────────────────┐
                │      Ollama LLM          │
                │ (Mistral / LLaMA 3)      │
                └─────────────┬───────────┘
                              │
                              ▼
                ┌─────────────────────────┐
                │   Final Answer to User   │
                └─────────────────────────┘

**Components**

1. Knowledge Source

-> Gale Encyclopedia of Medicine (Vol. 1).

-> OCR was applied to extract text from the scanned PDF.

2. Text Preprocessing and Embeddings

-> Text was split into chunks for better retrieval.

-> Embeddings were generated to represent text in vector space.

-> Embeddings enable semantic similarity matching.

2. Vector Database (Pinecone)

-> Pinecone was used as a managed vector database.

-> Stores embeddings and performs similarity search.

-> Returns the most relevant text passages based on the query.

3. Language Model Backend (Ollama)

-> Ollama was used to run local models.

-> Initially used TinyLlama due to hardware constraints.

-> Later tested with Mistral (7B) and LLaMA 3 (8B) on a system with 12 GB RAM.

-> The model generates answers only using the retrieved context.

4. Web Application (Flask)

-> Flask was used to build a simple web-based interface.

-> Users can enter medical queries and view responses in a chat-like format.

-> Flask handles routing, communication with Pinecone, and model interaction.


**Installation
**
1. Clone the repository

git clone https://github.com/yourusername/medical-chatbot.git
cd medical-chatbot


2. Create a virtual environment and install dependencies

python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt


3. Set up Pinecone

-> Sign up at https://app.pinecone.io

-> Create an index and get the API key

-> Set the API key in the environment:

set PINECONE_API_KEY=your_api_key   # Windows
export PINECONE_API_KEY=your_api_key   # Mac/Linux


4. Install Ollama and pull a model

5. ollama pull mistral


6. Run the Flask app

python app.py


**Usage**

-> Open the Flask app in the browser at http://127.0.0.1:5000/.

-> Enter a medical question, e.g.:

-> "What are the symptoms of asthma?"

-> "How is bronchitis treated?"

The chatbot retrieves relevant sections from the encyclopedia using Pinecone and generates a grounded response through the language model.

**Future Improvements**

-> Add more medical documents and journals to extend the knowledge base.

-> Enhance the UI for a more interactive experience.

-> Support speech input and output for accessibility.

-> Add evaluation metrics to measure medical accuracy.

-> Deploy on cloud infrastructure for multi-user access.
