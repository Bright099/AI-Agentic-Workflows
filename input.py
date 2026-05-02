from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools.retriever import create_retriever_tool
from langchain.agents import create_agent
load_dotenv()

model = init_chat_model(
    model = 'gemini-2.5-flash',
    model_provider = 'google_genai',
    temperature = 0.1
)


embeddings = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')

texts = [
    'I love apples.',
    'I enjoy oranges.',
    'I think pears taste very good.',
    'I hate bananas.',
    'I dislike raspberries.',
    'I hate mangos.',
    'I love Linux.',
    'I hate Windows.'
]

vector_store = FAISS.from_texts(texts, embedding=embeddings)

print(vector_store.similarity_search('what fruits does the person like?', k=3))
print(vector_store.similarity_search('What fruits does the person hate?', k=3))

retriever = vector_store.as_retriever(search_kwargs={'k': 6})
retriever_tool = create_retriever_tool(retriever, name='kb_search', description='Search the small product / tool knowledge base for information')

agent = create_agent(
    model = model,
    tools = [retriever_tool],
    system_prompt=(
        "You are a strict retrieval assistant. You HAVE NO personal knowledge of this user. You MUST call the kb_search tool for EVERY question. If the tool returns nothing, say you don't know. Do not guess."
    )
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What three fruits does the person like and what three fruits does the person dislike?"}]
})

print(result)
print(result["messages"][-1].content)