from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",

    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template="explain {topic} in detail and make a report",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="give me sort summary  of {report}, max 30 words",
    input_variables=["report"]
)

parser=StrOutputParser()

chain=prompt1|model|parser|prompt2|model|parser

result=chain.invoke({"topic": "blackhole"})

print(result)

chain.get_graph().print_ascii()