import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env")  # Point to your .env file if not in root

# Set up environment variables for Azure OpenAI
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_OPENAI_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION")

# Initialize Azure OpenAI Embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    openai_api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
)

# Load FAISS vector store
vectorstore_faiss = FAISS.load_local("faiss-db", embeddings, allow_dangerous_deserialization=True)

# Initialize Azure Chat OpenAI for LLM
llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    verbose=False,
    temperature=0.3,
)

# Define prompt template for the chatbot
PROMPT_TEMPLATE = """You are an AI Assistant who answers based on information in faiss-db. Given the following context:
{context}

Answer the following question:
{question}

Assistant:"""

PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

# Create a RetrievalQA instance
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 6}
    ),
    verbose=False,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
)

# Streamlit UI Setup
st.title("Azure OpenAI RAG Chatbot with FAISS")
st.write("Ask questions related to the FAISS document store.")

# Store chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# User question input
user_input = st.text_input("Ask a question:")

if st.button("Submit"):
    if user_input:
        # Retrieve response using QA chain
        response = qa.invoke({"query": user_input, "chat_history": st.session_state['chat_history']})
        
        # Display the answer
        st.write("**Answer:**", response["result"])
        
        # Append user query and response to chat history for follow-ups
        st.session_state['chat_history'].append((user_input, response["result"]))
        
        # Display the chat history (Q&A)
        st.write("### Chat History:")
        for i, (question, answer) in enumerate(st.session_state['chat_history']):
            st.write(f"**Q{i+1}:** {question}")
            st.write(f"**A{i+1}:** {answer}")
    else:
        st.warning("Please enter a question.")
