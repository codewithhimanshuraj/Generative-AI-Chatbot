from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model

model = init_chat_model("groq:openai/gpt-oss-120b")

response = model.invoke("write a poem on ai?")

print(response.content)
