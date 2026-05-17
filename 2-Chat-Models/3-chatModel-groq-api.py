from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

modle = ChatGroq(model="openai/gpt-oss-120b")

response = modle.invoke("what is the capital of pakistan")

print(response.content)