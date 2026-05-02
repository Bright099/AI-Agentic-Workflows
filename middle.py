from dataclasses import dataclass
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, dynamic_prompt

load_dotenv()

model = init_chat_model(
    model = 'gemini-2.5-flash',
    model_provider = 'google_genai',
    temperature = 0.1
)

@dataclass
class Context:
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.user_role

    base_prompt = 'You are a helpful and very concise assistant.'
    match user_role:
        case 'expert':
            return f'{base_prompt} Provide detail techical responses.'
        case 'beginner':
            return f'{base_prompt} Keep your explanations simple and basic.'
        case 'child':
            return f'{base_prompt} Explain everything like you were literally talking to a five-year old.'
        case _:
            return base_prompt

agent = create_agent(
    model = model,
    middleware = [user_role_prompt],
    context_schema = Context
)

response = agent.invoke({
    "messages": [{'role': 'user', 'content': 'Explain PCA.'}]
}, context=Context(user_role='expert'))

print(response)

