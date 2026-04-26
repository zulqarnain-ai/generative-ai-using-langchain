from  langchain_core.messages import SystemMessage, HumanMessage,AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline


llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = "text-generation",
    pipeline_kwargs=dict(
        do_sample=True,      
        temperature=0.7 

    )
)# type: ignore

model = ChatHuggingFace(llm=llm)

messages =[
    SystemMessage(content="you are a helpful assistant"),
    HumanMessage(content="tell me about lanchain")
]

result = model.invoke(messages)

messages.append(AIMessage(content = result.content))

print(messages)
