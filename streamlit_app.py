#OPENAI/NO_FC/Sales_Assist

import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader,Docx2txtLoader,TextLoader
from langchain.chains import RetrievalQA
import time
import sys
import os
import pickle
import asyncio

# Ensure an event loop is available and set it as the current event loop
loop = asyncio.get_event_loop_policy().new_event_loop()
asyncio.set_event_loop(loop)

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

llm = ChatOpenAI(
    model_name="gpt-4o",
    streaming=True
)
## ======================================
## Embedding
## ======================================
# chunk the data
def chunk_data(data, chunk_size):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(data)
    return chunks

# Create embeddings in chroma db
def create_embeddings(chunks):
    print("Embedding to Chroma DB...")
    #embeddings = QianfanEmbeddingsEndpoint()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large",disallowed_special=())
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings,persist_directory="./chroma_db")
    print("Done")
    return vector_store

def load_vector_store_from_file(file_path):
    with open(file_path, 'rb') as file:
        vector_store = pickle.load(file)
    return vector_store


## ======================================
## Promp template and question process
## ======================================
def ask_and_get_answer(vector_store, q):
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={'k': 3})
    prompt_template = """
    You are a professional trainer for the National Quality Framework (NQF). You are browsing the 'Guide to the NQF' document to answer questions related to the National Quality Framework. Please use the information from this guide to answer questions. If you don't know the answer, simply say "I don't know the answer to this question." Do not try to make up an answer. Do not use unprofessional terms; only respond in English.
    Manuals: {context}
    Question: {question}
    Answer:
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT,"verbose": True}
    chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs,verbose=True)
    result = chain.invoke(q)
    return result

# Simplify button clicks handling
def handle_button_click(question_key):
    st.session_state.clicked_question = question_key

def stream_data(answer):
    for word in answer.split():
        yield word + " "
        time.sleep(0.05)

# Process and display the question and answer
def process_question(question):
    # Add user message to chat history
    with st.chat_message("user"):
            st.markdown(question)
    st.session_state.history.append({"role": "user", "content": question})
    if "vector_store" in st.session_state:
        vector_store = st.session_state["vector_store"]
        answer = ask_and_get_answer(vector_store, question)
        #st.write(answer)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = st.write_stream(stream_data(answer['result']))
            print(response)
            result = None  # Default result to None to handle exceptions
            user_feedback=None
        # Add assistant response to chat history
        st.session_state.history.append({"role": "assistant", "content": response,"score":user_feedback,"image":result})
    # Reset clicked_question to prevent it from affecting subsequent actions
    st.session_state.clicked_question = None

# Main program
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)
    st.set_page_config(
    page_title="NFQ Training",
    page_icon="🏠",
    )

    st.subheader("NFQ Training")

    file = ["Explaining-the-amendments-to-the-National-Regulations-2020_0.pdf","Guide-to-the-NQF-230701a.pdf","Guide-to-the-NQF-changes-summary-table-September-2023_July.pdf"]

    if "data" not in st.session_state:
        #from langchain_community.document_loaders import PyPDFLoader
        from langchain_community.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loaders_list = [PyPDFLoader(pdf_file) for pdf_file in file]
        #loader = PyPDFLoader(file)
        from langchain_community.document_loaders.merge import MergedDataLoader
        loader = MergedDataLoader(loaders=loaders_list)
        st.session_state.data = loader.load() 
    
    # Define questions
    questions = {
        'question1': 'What are the key objectives of the National Quality Framework?',
        'question2': 'What are the seven quality areas outlined in the National Quality Standard?',
        'question3': 'How is the assessment and rating process conducted for education and care services?',
        'question4': 'What are the approved learning frameworks under the NQF?'
    }

    # Create buttons dynamically and handle clicks
    for question_key, question_text in questions.items():
        if st.button(question_text):
            handle_button_click(question_key)

    #st.write(st.session_state.data[0])
    if "chunks" not in st.session_state:
        st.session_state.chunks = chunk_data(st.session_state.data, 384)
    if "vector_store" not in st.session_state:
        if os.path.exists("./chroma_db"):
            st.session_state.vector_store = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings(model="text-embedding-3-large",disallowed_special=()))
        else:
            st.session_state.vector_store = create_embeddings(st.session_state.chunks)

    if "history" not in st.session_state:
        st.session_state.history = []

    # User input for the question
    # Display chat messages from history on app rerun
    for history in st.session_state.history:
        with st.chat_message(history["role"]):
            st.markdown(history["content"])

    # Check if a question was clicked
    if clicked_question := st.session_state.get('clicked_question'):
        process_question(questions[clicked_question])

    # Handle chat input
    if prompt := st.chat_input("Your Question:"):
# Ensure to reset any clicked question status before processing
        st.session_state.clicked_question = None
        process_question(prompt)