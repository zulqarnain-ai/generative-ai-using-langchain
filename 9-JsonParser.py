from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

model = ChatGroq(model = 'openai/gpt-oss-120b')
parser = JsonOutputParser()
template = PromptTemplate(
    input_variables=[],
    template='give me the name, age, and city of a fictional person \n {format_instruction}',
    partial_variables={'format_instruction':parser.get_format_instructions()}

)


chain = template | model | parser

result = chain.invoke({})
print(result)