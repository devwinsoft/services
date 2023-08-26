__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# from dotenv import load_dotenv
# load_dotenv();

import streamlit as st
import tempfile
import os
from langchain.document_loaders import PyPDFLoader

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(upload_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

st.title("OpenAI Test")
st.write("---")
upload_file = st.file_uploader("PDF 파일을 업로드 하세요.", type=['pdf'])
if upload_file is not None:
    # Load & Split
    pages = pdf_to_document(upload_file)
    text_splitter = RecursiveCharacterTextSplitter(        
        chunk_size = 300,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_documents(pages)

    st.write("---")
    question = st.text_input("질문을 해 주세요...")
    if st.button('요청'):
        with st.spinner('작성 중...'):
            # Embedding
            llm = ChatOpenAI(temperature=0)
            embeddings_model = OpenAIEmbeddings()
            db = Chroma.from_documents(texts, embeddings_model)
            retriever = MultiQueryRetriever.from_llm(retriever=db.as_retriever(), llm=llm)
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
            result = qa_chain({"query": question})
            st.write(result['result'])

# result = retriever.get_relevant_documents(query=question)
st.write("---")

# from streamlit_extras.buy_me_a_coffee import button
# button(username="maoshy", floating=True, width=220)
