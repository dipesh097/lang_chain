from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
print("-----",api_key[0:5],"******")


client=genai.Client(api_key=api_key)

response=client.models.generate_content(
    model="gemini-3.6-flash",
    contents="who is most famous person of india,give me top 5  name"
)

print(response.text)