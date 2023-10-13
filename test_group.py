llm_config = {"config_list": config_list_gpt4, "seed": 42}

ragproxyagent = RetrieveUserProxyAgent(
    name="ragproxyagent",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    retrieve_config={
        "task": "you need to fetch the answers related to company if needed",
        "docs_path": "/content/Docs",
        "chunk_token_size": 10,
        "model": config_list_gpt4[0]["model"],
        "client": chromadb.PersistentClient(path="/tmp/chromadb"),
        "embedding_model": "all-mpnet-base-v2",
    },
)
user_proxy = autogen.UserProxyAgent(
   name="User_proxy",
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=5,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={"work_dir": "web"},
    llm_config=llm_config,
    system_message="""You are a user of a mobile app. Reply to the agent with a suitable answer, otherwise reply I dont know or answer the agents questions with facts."""

)
support_agent = autogen.AssistantAgent(
    name="support_agent",
    llm_config=llm_config,
    system_message="You are the founder of a mobile app. You are trying to find the reason for less usage and uninstalls from the customer by asking relevant questions. \
    You try to answer the customer's query in the best possible way. If you do not know the answer ask for clarification or more data."
)
pm = autogen.AssistantAgent(
    name="PM",
    system_message="You are an expert in analysing if the objective of user research has been met from the conversation. Answer concisely.",
    llm_config=llm_config,
)
groupchat = autogen.GroupChat(agents=[user_proxy, support_agent, pm, ragproxyagent], messages=[], max_round=8)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)
but this seems not working the ragproxyagent is not participating in conversation