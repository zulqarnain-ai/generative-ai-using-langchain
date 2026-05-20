from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
model = ChatGroq(model="openai/gpt-oss-120b")

chat_prompt = ChatPromptTemplate([
    ('system',"you are a halpful customer support agent"),
    MessagesPlaceholder(variable_name="chat_history"),
    ('human',"{query}")
])

chats=[]
with open('chats.txt') as f:
    chats.extend(f.readlines())

prompt = chat_prompt.invoke({'chat_history':chats,'query':'where is my refund'})

response = model.invoke(prompt)

print(response.content)