from flask import Flask, render_template, request, jsonify
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize embeddings, loader, and vectorstore (global for efficiency)
embeddings = HuggingFaceEmbeddings()
loader = WebBaseLoader('https://www.ibm.com/think/topics/data-science')
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_documents = text_splitter.split_documents(docs)
vectors = FAISS.from_documents(final_documents, embeddings)

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name='Llama3-8b-8192')
prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the question on provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    Question: {input}
    """
)
document_chain = create_stuff_documents_chain(llm, prompt_template)
retriever = vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['user_input']
    if not user_input.strip():  # Ensure input is not empty
        return jsonify({"error": "Input cannot be empty."}), 400

    try:
        # Get the response from the chain
        response = retrieval_chain.invoke({"input": user_input})
        return jsonify({
            "answer": response['answer'],
            "context": [doc.page_content for doc in response['context']]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Dynamically get the port for deployment on platforms like Vercel or Render
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if PORT not set
    app.run(host='0.0.0.0', port=port, debug=True)  # Listen on all interfaces for cloud deployments
