from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

@tool
def multiply(a: int,b: int) -> int:
    """this tool take two numbers a and b and return their product"""
    return a*b

print(multiply.invoke({'a':3,'b':5}))
print(multiply.name)
print(multiply.description)
print(multiply.args)


# tool binding

model = ChatGroq(model="openai/gpt-oss-120b")

model_with_tool = model.bind_tools([multiply])

print('model output: ',model_with_tool.invoke('hi ').content)

# tool calling

query = HumanMessage('can you multiply 3 with 5 ')

masseges = [query]

result = model_with_tool.invoke(masseges)
 
masseges.append(result) # type: ignore

tool_message = multiply.invoke(result.tool_calls[0])

masseges.append(tool_message)

response = model_with_tool.invoke(masseges)

print('final result: ', response)