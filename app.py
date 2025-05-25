
from flask import Flask, render_template, request, jsonify
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

# Load Pinecone API Key
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
index_name = "medicalchatbot"
pc = Pinecone(api_key=PINECONE_API_KEY)

# Embeddings
embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Vector Store
vectorstore = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings
)

# Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# TinyLLaMA LLM
llm = Ollama(model="tinyllama")

# Anti-hallucination prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If the answer is not in the context, just say 'I don't know' and do not make anything up. "
    "Answer in three sentences maximum and keep the answer concise.\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Combine docs chain
qa_chain = create_stuff_documents_chain(llm, prompt)

# Retrieval chain
rag_chain = create_retrieval_chain(retriever, qa_chain)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '').strip()

    if not user_message:
        return jsonify({'response': "Please enter a valid question."})

    try:
        result = rag_chain.invoke({"input": user_message})
        response = result.get("answer", "").strip()

        if not response or "I don't" in response or "sorry" in response.lower():
            response = "I'm sorry, I don't have information on that topic in the book."

        return jsonify({'response': response})

    except Exception as e:
        print("Error:", e)
        return jsonify({'response': "Sorry, there was an error processing your request."})

if __name__ == '__main__':
    app.run(debug=True)

