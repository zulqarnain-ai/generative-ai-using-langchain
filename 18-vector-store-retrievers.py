from langchain_community.vectorstores import chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

documents  = [
    Document(page_content="LangChain helps developers build LLM application easily."),
    Document(page_content="chroma is a vector database optimized fot LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models.")
]

embadding_model = OpenAIEmbeddings()

vectorstore = chroma.from_documents( # type: ignore
    documents = documents,
    embedding = embadding_model,
    collection_name = "my_collection"
)

retriever = vectorstore.as_retriever(search_kwargs = {'k':2})

query = 'what is chroma used for'

results = retriever.invoke(query)
for i,doc in enumerate(results):
    print(f"\n =====Result {i+1}=====")
    print(doc.page_content)