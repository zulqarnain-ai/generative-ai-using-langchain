

import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# 1. Setup the Endpoint (Using Llama-3 which is standard for the API)
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text_generation",
    max_new_tokens=512,
    do_sample=False,
) # type: ignore

# 2. Wrap it in ChatHuggingFace
model = ChatHuggingFace(llm=llm)

# 3. Use HumanMessage objects (Best practice for ChatModels)
messages = [
    HumanMessage(content="What is the capital of Pakistan?")
]

try:
    result = model.invoke(messages)
    print(result.content)
except Exception as e:
    print(f"\n❌ Error still occurring: {e}")
    print("\n💡 Tip: Try using 'llm.invoke()' directly to see if the issue is with the Chat wrapper.")
