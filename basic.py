import requests
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage
load_dotenv()

model = init_chat_model(
    model = 'gemini-2.5-flash',
    model_provider = 'google_genai',
    temperature = 0.1
)

for chunk in model.stream('Hello, what is Python?'):
    print(chunk.text, end='', flush=True)