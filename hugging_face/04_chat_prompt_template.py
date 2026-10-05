  # chat prompt , there are all messages like system , human, ai messagge  Dynameic

from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

chat_model = ChatHuggingFace(llm=llm)
chat_template=ChatPromptTemplate(
    SystemMessage(content="you are exper in {domain}"),
    HumanMessage(content="Explain in simple terms , what is {topic}")
)

prompt=chat_template.invoke({"domain": "cricket","topic":"lpw"})

print(prompt)
