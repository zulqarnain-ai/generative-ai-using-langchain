from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b")

template = ChatPromptTemplate([
    ('system',"you are a {role} expert"),
    ('human', 'explain {topic} in simple terms under 5 lines')
])

prompt = template.invoke({'role': 'AI engineer','topic':'transformers'})

response = model.invoke(prompt)

print(response)