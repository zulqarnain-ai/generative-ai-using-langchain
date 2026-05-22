from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="openai/gpt-oss-120b",temperature=1)

class person(BaseModel):
    name: str = Field(description="name of the person")
    age:int = Field(gt=18, description='age of the person')
    city:str=Field(description='name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object=person)
template = PromptTemplate(
    template='write the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'place':'pakistan'})

print(result)