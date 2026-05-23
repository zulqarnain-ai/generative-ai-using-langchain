from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model =ChatGroq(model='openai/gpt-oss-120b')

promt = PromptTemplate(
    template='write a joke about {topic}',
    input_variables=['topic']
)

parser= StrOutputParser()

chain = RunnableSequence(promt, model, parser)

print(chain.invoke({'topic':'AI'}))