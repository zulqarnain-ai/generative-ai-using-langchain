print('libraries loaded...')
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.tools import InjectedToolArg
from typing import Annotated
import requests
import json
from dotenv import load_dotenv

load_dotenv()
print('libraries loaded')

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """Fetches the conversion rate between a base currency and a target currency."""
    url = f"https://v6.exchangerate-api.com/v6/cfe4fc5271168eac1e8dd4d7/pair/{base_currency}/{target_currency}"
    response = requests.get(url)
    return response.json()

@tool
def converter(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """Converts a base currency value to the target currency using the given conversion rate."""
    return base_currency_value * conversion_rate

# Tool binding
print('binding tools...')
model = ChatGroq(model="llama-3.3-70b-versatile")  # ✅ Fix 1: valid Groq model
model_with_tools = model.bind_tools([get_conversion_factor, converter])

messages = [HumanMessage("What is the conversion factor between USD and PKR, and based on that can you convert 20 USD to PKR?")]

print('invoking model...')
ai_message = model_with_tools.invoke(messages)
print('\n\nmodel response: ', ai_message)
messages.append(ai_message)  # type: ignore

# ✅ Fix 3: Process get_conversion_factor calls first, then converter
conversion_rate = None

# First pass: handle get_conversion_factor
for tool_call in ai_message.tool_calls:
    if tool_call['name'] == 'get_conversion_factor':
        print('Calling get_conversion_factor...')
        tool_message1 = get_conversion_factor.invoke(tool_call)
        messages.append(tool_message1)

        # ✅ Fix 4: safely extract conversion_rate
        try:
            data = json.loads(tool_message1.content)
            conversion_rate = data.get('conversion_rate')
            print(f'conversion_rate fetched: {conversion_rate}')
        except (json.JSONDecodeError, KeyError) as e:
            print(f'Error parsing conversion rate: {e}')

# Second pass: handle converter
for tool_call in ai_message.tool_calls:
    if tool_call['name'] == 'converter':  # ✅ Fix 2: was `tool_call == 'converter'`
        print('Calling converter...')
        if conversion_rate is not None:
            tool_call['args']['conversion_rate'] = conversion_rate
            tool_message2 = converter.invoke(tool_call)
            messages.append(tool_message2)
        else:
            print('Skipping converter: conversion_rate not available.')

response = model_with_tools.invoke(messages)
print(response)
print('\nFinal response:', response.content)