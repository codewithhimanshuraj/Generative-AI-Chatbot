from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

model = init_chat_model("groq:openai/gpt-oss-120b")

print("Choose your AI mode")
print("Press 1 for angry mode")
print("Press 2 for funny mode")
print("Press 3 for sad mode")

choice = int(input("Tell your response: "))

if choice == 1:
    mode = "You are an angry AI agent. Respond aggressively and impatiently."

elif choice == 2:
    mode = "You are a funny AI agent. Respond humorously and playfully."

elif choice == 3:
    mode = "You are a sad AI agent. Respond empathetically and compassionately."

else:
    print("Invalid choice!")
    exit()


messages = [
    SystemMessage(content=mode),
]

print("------------ Welcome! Type 0 to exit the application ------------")

while True:
    prompt = input("You : ")

    if prompt == "0":
        break

    messages.append(HumanMessage(content=prompt))

    response = model.invoke(messages)

    messages.append(AIMessage(content=response.content))

    print("Bot :", response.content)