from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence,RunnableParallel

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template="generate a tweet on {topic}",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="generate a linkedin post on {topic}",
    input_variables=["topic"]
)

parser=StrOutputParser()

parallel_chain=RunnableParallel(
    {
     "tweet":RunnableSequence(prompt1,model,parser),
     "linkedin":RunnableSequence(prompt2,model,parser)
    }
)

result=parallel_chain.invoke({"topic":"GenAI"})

print(result)