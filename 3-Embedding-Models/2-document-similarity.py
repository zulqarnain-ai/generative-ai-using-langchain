from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

document = [
    "The cat is sitting on the mat.",
    "A cat is resting on a mat.",
    "There is a cat lying on the mat.",
    "The mat has a cat sitting on it.",
    "The student is studying for the final exam.",
    "A learner is preparing for their exams.",
    "He is revising his course material for the test.",
    "She is getting ready for her final assessment.",
    "Artificial intelligence is transforming modern industries.",
    "AI is changing how industries operate today.",
    "Machine learning is revolutionizing businesses.",
    "Technology driven by AI is reshaping the world.",
]

text ="what reshaping the world"

document_embedding = embedding.embed_documents(document)
query_embedding = embedding.embed_query(text)

scores = cosine_similarity([query_embedding],document_embedding)

max_value = np.max(scores)

max_index = np.argmax(scores)

print(text)
print (document[max_index])
print("AI confidence: ",max_value*100,"%")