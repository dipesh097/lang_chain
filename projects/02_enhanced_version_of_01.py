
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores  import FAISS
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()


st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📚",
    layout="wide"
)

# ------------------------------------------------------------------------------------------------
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# ------------------------------------------------------------------------------------------------

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

st.title("📚 AI Research Paper Assistant")
st.write("upload a pdf and ask quetion about it")

file=st.file_uploader("Upload your pdf....",["pdf"])

parser=StrOutputParser()

if file is not None:
    with open("temp2.pdf","wb") as f:
        f.write(file.getbuffer())

    loader=PyPDFLoader("temp2.pdf")
    document=loader.load()

    print("------------------------------------------------------------------------------",document[0].metadata)

    splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks=splitter.split_documents(document)

    col1,col2=st.columns(2)

    with col1:
        st.metric("no. of pages in document",len(document))
    with col2:
        st.metric("no. of chunks ",len(chunks))


    user_query=st.text_area("enter your query related to uploaded pdf :",
                            placeholder="what is main idea of document")

    if st.button("generate"):
        vector_store=FAISS.from_documents(chunks,embedding)

        docs=vector_store.similarity_search(user_query,k=3)

        context=[]
        no_page=[]
        for doc in docs:
            context.append(doc.page_content)

            no_page.append(doc.metadata["page"])

        context="\n\n".join(context)

        print(no_page)


        prompt=PromptTemplate(
            template="""ROLE
    You are an AI research assistant.

    CONTEXT
   {context}

   QUESTION
   {query}

   INSTRUCTION
   Answer using ONLY the context.
   If the answer isn't there, say you couldn't find it.""",
        input_variables=["context","query"]
        )

        chain=prompt|model|parser

        st.write(chain.invoke({"query":user_query,"context":context}))
        st.write("-------------------------------")
        for page in no_page:
             st.text("sources:",page)















