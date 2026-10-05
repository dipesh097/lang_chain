from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)


prompt1=PromptTemplate(
    template="write a  5 lines poem on  \n {topic} in hinglish",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="explain this poem {poem}",
    input_variables=["poem"]
)

parser=StrOutputParser()

joke_gen_chain=RunnableSequence(prompt1,model,parser)

parallel_chain=RunnableParallel(
    { "joke":RunnablePassthrough(),
      "explanatioin":RunnableSequence(prompt2,model,parser)  })

final_chain=RunnableSequence(joke_gen_chain,parallel_chain)

print(final_chain.invoke({"topic": "funny"}))