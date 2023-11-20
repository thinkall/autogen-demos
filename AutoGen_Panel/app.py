import os
import time

import autogen
import panel as pn
import param
from autogen_utils import initialize_agents, thread_with_trace
from panel.chat import ChatInterface
from panel.widgets import Button, PasswordInput, Select, TextInput

TIMEOUT = 60
pn.extension()


def get_description_text():
    return """
    # Microsoft AutoGen: Playground

    This is an AutoGen playground.

    #### [[AutoGen](https://github.com/microsoft/autogen)] [[Discord](https://discord.gg/pAbnFJrkgZ)] [[Paper](https://arxiv.org/abs/2308.08155)] [[SourceCode](https://github.com/thinkall/autogen-demos)]
    """


pn.pane.Markdown(get_description_text(), sizing_mode="stretch_width").servable()

txt_model = TextInput(
    name="Model Name", placeholder="Enter your model name here...", value="gpt-35-turbo", sizing_mode="stretch_width"
)
pwd_openai_key = PasswordInput(
    name="OpenAI API Key", placeholder="Enter your OpenAI API Key here...", sizing_mode="stretch_width"
)
pwd_aoai_key = PasswordInput(
    name="Azure OpenAI API Key", placeholder="Enter your Azure OpenAI API Key here...", sizing_mode="stretch_width"
)
pwd_aoai_url = PasswordInput(
    name="Azure OpenAI Base Url", placeholder="Enter your Azure OpenAI Base Url here...", sizing_mode="stretch_width"
)
pn.Row(txt_model, pwd_openai_key, pwd_aoai_key, pwd_aoai_url).servable()


def get_config():
    config_list = autogen.config_list_from_json(
        "OAI_CONFIG_LIST",
        file_location=".",
    )
    if not config_list:
        os.environ["MODEL"] = txt_model.value
        os.environ["OPENAI_API_KEY"] = pwd_openai_key.value
        os.environ["AZURE_OPENAI_API_KEY"] = pwd_aoai_key.value
        os.environ["AZURE_OPENAI_API_BASE"] = pwd_aoai_url.value

        config_list = autogen.config_list_from_models(
            model_list=[os.environ.get("MODEL", "gpt-35-turbo")],
        )
    if not config_list:
        config_list = [
            {
                "api_key": "",
                "base_url": "",
                "api_type": "azure",
                "api_version": "2023-07-01-preview",
                "model": "gpt-35-turbo",
            }
        ]

    llm_config = {
        "timeout": 60,
        "cache_seed": 42,
        "config_list": config_list,
        "temperature": 0,
    }

    return llm_config


btn_add = Button(name="+", button_type="success")
btn_remove = Button(name="-", button_type="danger")
pn.Row(
    pn.pane.Markdown("## Add or Remove Agents: "),
    btn_add,
    btn_remove,
).servable()

row_agent = """
pn.Row(
    TextInput(name="", value={agent_name}, placeholder="Agent Name", min_width=100, sizing_mode="scale_width"),
    TextInput(name="", value={system_msg}, placeholder="System Message", min_width=400, sizing_mode="scale_width"),
    Select(
        name="",
        value={agent_type},
        min_width=100,
        options=[
            "AssistantAgent",
            "UserProxyAgent",
            "RetrieveUserProxyAgent",
            "RetrieveAssistantAgent",
            "CompressibleAgent",
            "GPTAssistantAgent",
            "LLaVAAgent",
            "MathUserProxyAgent",
            "TeachableAgent",
        ],
        sizing_mode="scale_width",
    ),
)
"""

column_agents = pn.Column(
    eval(
        row_agent.format(
            agent_name="'Boss'",
            system_msg="'The boss who ask questions and give tasks. Reply `TERMINATE` if the task is done.'",
            agent_type="'UserProxyAgent'",
        )
    ),
    sizing_mode="stretch_width",
)
column_agents.append(
    eval(
        row_agent.format(
            agent_name="'Senior_Python_Engineer'",
            system_msg="'You are a senior python engineer. Reply `TERMINATE` if the task is done.'",
            agent_type="'AssistantAgent'",
        )
    )
)
column_agents.append(
    eval(
        row_agent.format(
            agent_name="'Product_Manager'",
            system_msg="'You are a product manager. Reply `TERMINATE` if the task is done.'",
            agent_type="'AssistantAgent'",
        )
    )
)

column_agents.servable()


def add_agent(event):
    column_agents.append(
        eval(
            row_agent.format(
                agent_name="''",
                system_msg="''",
                agent_type="'AssistantAgent'",
            )
        )
    )


def remove_agent(event):
    column_agents.pop(-1)


btn_add.on_click(add_agent)
btn_remove.on_click(remove_agent)


def send_messages(recipient, messages, sender, config):
    chatiface.send(messages[-1]["content"], user=messages[-1]["name"], respond=False)
    return False, None  # required to ensure the agent communication flow continues


def init_groupchat(event):
    llm_config = get_config()
    agents = []
    for row_agent in column_agents:
        agent_name = row_agent[0].value
        system_msg = row_agent[1].value
        agent_type = row_agent[2].value
        agent = initialize_agents(llm_config, agent_name, system_msg, agent_type)
        agent.register_reply([autogen.Agent, None], reply_func=send_messages, config={"callback": None})
        agents.append(agent)

    groupchat = autogen.GroupChat(agents=agents, messages=[], max_round=12, speaker_selection_method="round_robin")
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)
    return agents, manager


def get_num(contents, user, instance):
    num = len(contents)
    history = [message for message in instance.objects]
    history = [message.object for message in history if message.user != "System"]
    return f"Got {num}. {user=}. {history=}. {pwd_openai_key.value=}. {[agent[0].value for agent in column_agents]=}."


def agents_chat(init_sender, manager, contents):
    init_sender.initiate_chat(manager, message=contents)


def agents_chat_thread(init_sender, manager, contents):
    """Chat with the agent through terminal."""
    thread = thread_with_trace(target=agents_chat, args=(init_sender, manager, contents))
    thread.start()
    for i in range(int(TIMEOUT * 10)):
        thread.join()
        time.sleep(0.1)
    try:
        thread.join()
        if thread.is_alive():
            thread.kill()
            thread.join()
            chatiface.send("Timeout Error: Please check your API keys and try again later.")
    except Exception as e:
        chatiface.send(str(e) if len(str(e)) > 0 else "Invalid Request to OpenAI, please check your API keys.")


def reply_chat(contents, user, instance):
    [message for message in instance.objects]
    if hasattr(instance, "agents"):
        agents = instance.agents
        manager = instance.manager
    else:
        agents, manager = init_groupchat(None)
        instance.manager = manager
        instance.agents = agents
    init_sender = None
    for agent in agents:
        if "UserProxy" in str(type(agent)):
            init_sender = agent
            break
    if not init_sender:
        init_sender = agents[0]
    agents_chat(init_sender, manager, contents)


chatiface = ChatInterface(
    callback=reply_chat,
    height=600,
)

chatiface.servable()
