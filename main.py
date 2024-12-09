import src.base_chat as base_chat

model = base_chat.Base_Chat()
print("AI: Hello.")
while True:
    user_input = input("User: ")
    response = model.chat_with_history(user_input)
    