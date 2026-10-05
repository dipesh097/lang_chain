
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

import streamlit as st

loader=PyPDFLoader(r"C:\Users\dipesh's-device\Documents\SIH2026-aaditya.pdf")

file=loader.load()

# file=st.file_uploader("upload your file",["pdf"])
query=st.text_input("enter your query :")



if file is not None :
    # with open("temp.pdf", "wb") as f:
    #     f.write(file.getbuffer())

    # loader=PyPDFLoader("temp.pdf")

    document=loader.load()

    # print(document[2].page_content)

    spilitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks=spilitter.split_documents(document)

    print(len(chunks))

    embedding=HuggingFaceEmbeddings(
       model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector=embedding.embed_query(chunks[0].page_content)


    print(len(vector))


    vector_strore=FAISS.from_documents(chunks,embedding)



    if query:
        docs=vector_strore.similarity_search(query,k=3)

        # for i , doc in enumerate(docs):
        #   print(i,doc)


        load_dotenv()

        llm=HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.1-8B-Instruct",
            task="text-generation"
        )

        model=ChatHuggingFace(llm=llm)

        context="\n\n".join(doc.page_content for doc in docs)
        content = []
        page = []
        for doc in docs:
            content.append(doc.page_content)

            page.append(doc.metadata["page"])

        print(page)
        print("------------------------------------------------------------------------------------------------------")
# print(context)


        prompt=PromptTemplate(
            template="""
            You are an AI research assistant.

        Answer the question using ONLY the provided context.

        If the answer is not present in the context, say:
        "I could not find this information in the paper."
         {context} And {query}
        """ ,
        input_variables=["context","query"]
        )

        final_prompt=prompt.invoke(
            {"context": context,
             "query":query
             }
        )

        print(model.invoke(final_prompt).content)

        #
        # if st.button("generate"):
        #     result = model.invoke(final_prompt)
        #     st.write(result.content)


