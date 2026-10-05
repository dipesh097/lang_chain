from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda
from pydantic import BaseModel,Field
from typing import Literal


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)


parser1=StrOutputParser()

class feedback(BaseModel):
    sentiment:Literal["positive","negative"]=Field(description="give me sentiment of given feedback")

parser2=PydanticOutputParser(pydantic_object=feedback)

prompt1=PromptTemplate(
    template="analyze this following text and tell me sentiment \n {text} \n {format_instruction}",
    input_variables=["text"],
    partial_variables={"format_instruction":parser2.get_format_instructions()}
)

classifier_chain=prompt1|model|parser2

# result=classifier_chain.invoke({"text":"this is very bad product"})
# print(result)

prompt2=PromptTemplate(
    template="write a appropiate response for this positive response \n {feedback}",
    input_variables=["feedback"]
)

prompt3=PromptTemplate(
    template="write a appropiate response for this negative response \n {feedback}",
    input_variables = ["feedback"]
)

branch_chain=RunnableBranch(
    # (condition,chain)
    (lambda x:x.sentiment=="positive",prompt2|model|parser1),
         (lambda x:x.sentiment=="negative",prompt3|model|parser1),
           RunnableLambda(lambda x : "could not find any sentiment")

          )

chain=classifier_chain|branch_chain

print(chain.invoke({"text":"this is not worth it bro !"}))