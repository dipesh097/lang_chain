from langchain_google_genai import ChatGoogleGenerativeAI
import os

from dotenv import load_dotenv

load_dotenv()

api_key=os.getenv("GOOGLE_API_KEY")

llm=ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key
)

response=llm.invoke(
    "which ai model is currently most famous and new.give me only one name"
)

print(response.text)