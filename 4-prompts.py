from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()
topic = input("enter topic: ")
language = input("select language: ")

prompt = PromptTemplate(
    input_variables = ['topic','language'],
    template ="Explain {topic} in simple terms and in {language} language and also explanation should be less then five lines"
    )
formeted = prompt.format(topic = topic, language = language)
print(formeted)
model = ChatGroq(model="openai/gpt-oss-120b")

result = model.invoke(formeted)

print("result: ", result.content)