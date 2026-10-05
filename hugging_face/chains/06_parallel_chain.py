from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)


prompt1=PromptTemplate(
    template="generate sort quetions on following text \n {text}",
    input_variables=["text"]
)

prompt2=PromptTemplate(
    template="generate a sort quetion answer quiz on {text}",
    input_variables=["text"]
)

prompt3=PromptTemplate(
    template="now combine s both notes and quetion-answer {notes}, {quiz}, so i can easy understand topic",
    input_variables=["notes","quiz"]
)

parser=StrOutputParser()

parallel_chain=RunnableParallel(
    {"notes":prompt1|model|parser,
     "quiz":prompt2|model|parser
     }
)

merging_chain=prompt3|model|parser

complete_chain=parallel_chain|merging_chain

result=complete_chain.invoke({"text":"blackhole"})

print(result)

complete_chain.get_graph().print_ascii()