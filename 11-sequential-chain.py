from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model = "openai/gpt-oss-120b")

prompt1 = PromptTemplate(
    input_variables=['topic'],
    template="write a detailed report on {topic}"
)

prompt2 = PromptTemplate(
    input_variables=['text'],
    template="give me an exact 5 points from the following text \n {text}"
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result =chain.invoke({'topic':'Transformer'})

print(result)