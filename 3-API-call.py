from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatGroq(model='openai/gpt-oss-120b')

message = HumanMessage(
    content='what is the capital of pakistan'
)

respons = llm.invoke([message])

print("response: ",respons.content)