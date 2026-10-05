from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnableLambda,RunnableBranch,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)


prompt1=PromptTemplate(
    template="generate a detailed report on {topic}",
    input_variables=["topic"]
)

prompt2=PromptTemplate(
    template="summarize this {report}",
    input_variables=["report"]
)

parser=StrOutputParser()

report_generator=RunnableSequence(prompt1,model,parser)

branch_runnable=RunnableBranch(
    (lambda x: len(x.split())<500,RunnablePassthrough()),
     (lambda x:len(x.split())>500,RunnableSequence(prompt2,model,parser)),
     RunnableLambda(lambda x:"could not get any input")
     )

chain=RunnableSequence(report_generator,branch_runnable)

parallel_chain=RunnableParallel(
    {
        "report":RunnablePassthrough(),
        "word_count":RunnableLambda(lambda x:len(x.split()))
    }
)
final_chain=RunnableSequence(chain,parallel_chain)
print(final_chain.invoke({"topic": "boom og Agentic Ai"}))


