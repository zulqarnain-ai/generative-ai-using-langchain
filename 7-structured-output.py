from langchain_groq import ChatGroq
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

# CRITICAL: gpt-oss-120b requires temperature=1.0 on Groq for reliable tool use
model = ChatGroq(
    model='openai/gpt-oss-120b',
    temperature=1.0
)

class Review(TypedDict):
    summary: str  # Fixed typo here
    sentiment: str

structured_model = model.with_structured_output(Review) # type: ignore

# Explicitly framing the text so the reasoning model doesn't drift into conversational mode
prompt = """Analyze the following product review and extract the summary and sentiment fields as requested by your tool format:

Review: "I was really looking forward to using this, but it stopped working after three days. The material feels very flimsy compared to the pictures. Customer service was unhelpful. I would not recommend."
"""

result = structured_model.invoke(prompt)

print(result) # type: ignore
