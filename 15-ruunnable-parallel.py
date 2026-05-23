from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model1 = ChatGroq(model='openai/gpt-oss-120b')
model2 = ChatGroq(model='llama-3.3-70b-versatile')

prompt1 = PromptTemplate(
    template="give me a tweet about {topic}",
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='generate a LinkedIn post about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        'tweet': prompt1 | model1 | parser,
        'post': prompt2 |model2 |parser
    }
)

result = parallel_chain.invoke({'topic':"AI"})

print(result['tweet'])
print()
print(result['post'])