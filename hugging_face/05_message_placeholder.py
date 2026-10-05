from langchain_core import chat_history
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

chat_template=ChatPromptTemplate( [
    ("system","you are a helpfull AI assistent"),
    MessagesPlaceholder(variable_name="chat_history")
])

