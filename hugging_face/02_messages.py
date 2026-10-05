from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
chat_model=ChatHuggingFace(llm=llm)
memory=[SystemMessage(content="you ae a professional ai Assistent ")]
while True:
     user_input=str(input("enter your query 😎 :"))

     if user_input.lower()== "exit" :
         print("thankyou")
         break
     memory.append(HumanMessage(user_input))

     result=chat_model.invoke(memory)

     memory.append(AIMessage(result.content))

     print(result.content)

print(memory)
