import os

from google import genai
from dotenv import load_dotenv

load_dotenv()

api_key=os.getenv("GOOGLE_API_KEY")

if api_key:
    print("api key loaded successfully ")
else :
    print("😒,you are not able to do this okay")

client=genai.Client(api_key=api_key)

while True:
    user_input=input("ask me any quetions about anything :")

    response=client.models.generate_content(
        model="gemini-3.6-flash",
        contents=user_input
    )

    print(response.text)
