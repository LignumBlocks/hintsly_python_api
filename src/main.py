# import chatbot.base_chat as base_chat

# model = base_chat.Base_Chat()
# print("AI: Hello.")
# while True:
#     user_input = input("User: ")
#     response = model.chat_with_history(user_input)

import superhacks.generate_superhacks as sh
import utils.vector_store_management as vsm
import utils.handle_hintsly_api as api
sh.pipeline1()
# api.main()
# vs_m = vsm.VS_Manager()
# print(vs_m.get_by_ids(['5859', '5601', '5736', '5755']))
