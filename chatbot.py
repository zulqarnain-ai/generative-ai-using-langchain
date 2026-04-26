from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import HumanMessage, AIMessage
from transformers import AutoTokenizer, pipeline

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    return_full_text=False,
    # max_new_tokens=100, 
    do_sample=True, 
    temperature=0.1, # Lower temperature makes the model less "random"/crazy
    top_p=0.9,
)

llm = HuggingFacePipeline(pipeline=pipe)
model = ChatHuggingFace(llm=llm)

# Use a list of Message objects for history
chat_history = [] 

print("Chatbot is ready! type 'exit' to quit.")

while True:
    user_input = input('You: ')
    if user_input.lower() == 'exit':
        break
    
    # 1. Add user message to history
    chat_history.append(HumanMessage(content=user_input))

    # 2. Pass the ENTIRE history so it has context
    result = model.invoke(chat_history)
    
    # 3. Add AI response to history
    chat_history.append(AIMessage(content=result.content))
    
    print('AI:', result.content.strip()) # type: ignore