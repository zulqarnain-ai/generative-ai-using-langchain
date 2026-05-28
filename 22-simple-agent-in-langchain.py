from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.tools import tool

# Define a simple search tool
@tool
def search(query: str) -> str:
    """Search for information about the query"""
    # Placeholder implementation - you can replace with actual search
    return f"Search results for: {query}"

tools = [search]

model = ChatGroq(model="llama-3.3-70b-versatile")

# Create agent with new API
agent = create_agent(
    model=model,
    tools=tools,
    system_prompt="You are a helpful assistant that can search for information.",
)

# Execute agent
response = agent.invoke({
    'messages': [{'role': 'user', 'content': '3 ways to reach lahore from Islamabad'}]
})
print(response)

