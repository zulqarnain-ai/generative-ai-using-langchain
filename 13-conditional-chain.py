from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch,RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model= "openai/gpt-oss-120b")

strparser = StrOutputParser()

class feadback(BaseModel):

    sentiment:Literal['positive', 'negative'] = Field(description="give me the sentiment of the feadback")

pyparser = PydanticOutputParser(pydantic_object=feadback)

prompt1 = PromptTemplate(
    template='classify the sentiment of the following feadback text into positive and negative \n {text} \n {format_instruction}',
    input_variables=['text'],
    partial_variables=({'format_instruction':pyparser.get_format_instructions()})
)
prompt2 =PromptTemplate(
    template='write an apropriate response to this positive feadback \n {feadback}',
    input_variables=['feadback']
)
prompt3 =PromptTemplate(
    template='write an apropriate response to this negative feadback \n {feadback}',
    input_variables=['feadback']
)

chain1 = prompt1 | model | pyparser

chain2 = RunnableBranch(
    (lambda x:x.sentiment =='positive', prompt2 | model | strparser), # type: ignore
    (lambda x:x.sentiment =='negative', prompt3 | model | strparser), # type: ignore
    RunnableLambda(lambda x:'sentiment not fount')
)

chain = chain1 | chain2

print(chain.invoke({'text':'this is a beautiful phone'}))
