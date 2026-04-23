from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = "text-generation",
    pipeline_kwargs=dict(
        max_new_tokens = 100,
        do_sample=True,      
        temperature=0.7 

    )
)# type: ignore

model = ChatHuggingFace(llm=llm)

result = model.invoke('what is the capital of pakistan and the population of the capital')

print(result.content)