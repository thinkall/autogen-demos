import asyncio
import os
import random
import time
from functools import partial

import autogen
import panel as pn
from autogen_utils import (
    MathUserProxyAgent,
    RetrieveUserProxyAgent,
    check_termination_and_human_reply,
    generate_code,
    get_retrieve_config,
    initialize_agents,
)
from configs import DEFAULT_TERMINATE_MESSAGE, Q1, Q2, Q3, TIMEOUT, TITLE
from custom_widgets import RowAgentWidget
from panel.chat import ChatInterface
from panel.widgets import Button, CodeEditor, PasswordInput, Switch, TextInput

pn.extension("codeeditor")

template = pn.template.BootstrapTemplate(title=TITLE)


def get_description_text():
    return f"""
    # {TITLE}

    This is an AutoGen playground built with [Panel](https://panel.holoviz.org/). You can use it to interact with the AutoGen agents. Scroll down to see the code for creating and using the agents.

    #### [[AutoGen](https://github.com/microsoft/autogen)] [[Discord](https://discord.gg/pAbnFJrkgZ)] [[Paper](https://arxiv.org/abs/2308.08155)] [[SourceCode](https://github.com/thinkall/autogen-demos)]
    """


template.main.append(pn.pane.Markdown(get_description_text(), sizing_mode="stretch_width"))

txt_model = TextInput(
    name="Model Name", placeholder="Enter your model name here...", value="gpt-35-turbo", sizing_mode="stretch_width"
)
pwd_openai_key = PasswordInput(
    name="OpenAI API Key", placeholder="Enter your OpenAI API Key here...", sizing_mode="stretch_width"
)
pwd_openai_url = PasswordInput(
    name="OpenAI Base Url", placeholder="Enter your OpenAI Base Url here...", sizing_mode="stretch_width"
)
pwd_aoai_key = PasswordInput(
    name="Azure OpenAI API Key", placeholder="Enter your Azure OpenAI API Key here...", sizing_mode="stretch_width"
)
pwd_aoai_url = PasswordInput(
    name="Azure OpenAI Base Url", placeholder="Enter your Azure OpenAI Base Url here...", sizing_mode="stretch_width"
)
file_cfg = pn.widgets.FileInput(filename="OAI_CONFIG_LIST", sizing_mode="stretch_width")
template.main.append(pn.Row(txt_model, pwd_openai_key, pwd_openai_url, pwd_aoai_key, pwd_aoai_url, file_cfg))


def get_config(tmpfilename="OAI_CONFIG_LIST"):
    os.makedirs(".chromadb", exist_ok=True)
    if file_cfg.value:
        if os.path.exists(f".chromadb/{tmpfilename}"):
            os.remove(f".chromadb/{tmpfilename}")
        file_cfg.save(f".chromadb/{tmpfilename}")
        cfg_fpath = f".chromadb/{tmpfilename}"
    else:
        cfg_fpath = "OAI_CONFIG_LIST"  # for local testing
    config_list = autogen.config_list_from_json(
        cfg_fpath,
        file_location=".",
    )
    if not config_list:
        os.environ["MODEL"] = txt_model.value
        os.environ["OPENAI_API_KEY"] = pwd_openai_key.value
        os.environ["OPENAI_API_BASE"] = pwd_openai_url.value
        os.environ["AZURE_OPENAI_API_KEY"] = pwd_aoai_key.value
        os.environ["AZURE_OPENAI_API_BASE"] = pwd_aoai_url.value

        config_list = autogen.config_list_from_models(
            model_list=[os.environ.get("MODEL", "gpt-35-turbo")],
        )
        for cfg in config_list:
            if cfg.get("api_type", "open_ai") == "open_ai":
                base_url = os.environ.get("OPENAI_API_BASE", "").strip()
                if base_url:
                    cfg["base_url"] = base_url
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
        "timeout": TIMEOUT,
        "cache_seed": 42,
        "config_list": config_list,
        "temperature": 0,
    }

    return llm_config


btn_add = Button(name="+", button_type="success")
btn_remove = Button(name="-", button_type="danger")
switch_code = Switch(name="Run Code", sizing_mode="fixed", width=50, height=30, align="end")
select_speaker_method = pn.widgets.Select(name="", options=["round_robin", "auto", "random"], value="round_robin")
template.main.append(
    pn.Row(
        pn.pane.Markdown("## Add or Remove Agents: "),
        btn_add,
        btn_remove,
        pn.pane.Markdown("### Run Code: "),
        switch_code,
        pn.pane.Markdown("### Speaker Selection Method: "),
        select_speaker_method,
    )
)

column_agents = pn.Column(
    RowAgentWidget(
        value=[
            "User_Proxy",
            "",
            "UserProxyAgent",
            "",
        ]
    ),
    sizing_mode="stretch_width",
)
column_agents.append(
    RowAgentWidget(
        value=[
            "Assistant_Agent",
            "",
            "AssistantAgent",
            "",
        ]
    ),
)

template.main.append(column_agents)


def add_agent(event):
    column_agents.append(RowAgentWidget(value=["", "", "AssistantAgent", ""]))


def remove_agent(event):
    column_agents.pop(-1)


btn_add.on_click(add_agent)
btn_remove.on_click(remove_agent)


async def send_messages(recipient, messages, sender, config):
    # print(f"{sender.name} -> {recipient.name}: {messages[-1]['content']}")
    chatiface.send(messages[-1]["content"], user=sender.name, respond=False)
    return False, None  # required to ensure the agent communication flow continues


class myGroupChatManager(autogen.GroupChatManager):
    def _send_messages(self, message, sender, config):
        message = self._message_to_dict(message)

        if message.get("role") == "function":
            content = message["content"]
        else:
            content = message.get("content")
            if content is not None:
                if "context" in message:
                    content = autogen.OpenAIWrapper.instantiate(
                        content,
                        message["context"],
                        self.llm_config and self.llm_config.get("allow_format_str_template", False),
                    )
            if "function_call" in message:
                function_call = dict(message["function_call"])
                content = f"Suggested function Call: {function_call.get('name', '(No function name found)')}"
        chatiface.send(content, user=sender.name, respond=False)
        return False, None  # required to ensure the agent communication flow continues

    def _process_received_message(self, message, sender, silent):
        message = self._message_to_dict(message)
        # When the agent receives a message, the role of the message is "user". (If 'role' exists and is 'function', it will remain unchanged.)
        valid = self._append_oai_message(message, "user", sender)
        if not valid:
            raise ValueError(
                "Received message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided."
            )
        if not silent:
            self._print_received_message(message, sender)
            self._send_messages(message, sender, None)


def init_groupchat(event, collection_name):
    llm_config = get_config(collection_name)
    agents = []
    for row_agent in column_agents:
        agent_name = row_agent[0][0].value
        system_msg = row_agent[0][1].value
        agent_type = row_agent[0][2].value
        docs_path = row_agent[1].value
        retrieve_config = (
            get_retrieve_config(
                docs_path,
                txt_model.value,
                collection_name=collection_name,
            )
            if agent_type == "RetrieveUserProxyAgent"
            else None
        )
        code_execution_config = (
            {
                "work_dir": "coding",
                "use_docker": False,  # set to True or image name like "python:3" to use docker
            }
            if switch_code.value
            else False
        )
        agent = initialize_agents(
            llm_config, agent_name, system_msg, agent_type, retrieve_config, code_execution_config
        )
        agent.register_reply([autogen.Agent, None], reply_func=send_messages, config={"callback": None})
        agents.append(agent)
    if len(agents) >= 3:
        groupchat = autogen.GroupChat(
            agents=agents,
            messages=[],
            max_round=12,
            speaker_selection_method=select_speaker_method.value,
            allow_repeat_speaker=False,
        )
        manager = myGroupChatManager(groupchat=groupchat, llm_config=llm_config)
    else:
        manager = None
        groupchat = None
    return agents, manager, groupchat


async def agents_chat(init_sender, manager, contents, agents):
    recipient = manager if len(agents) > 2 else agents[1] if agents[1] != init_sender else agents[0]
    if isinstance(init_sender, (RetrieveUserProxyAgent, MathUserProxyAgent)):
        await init_sender.a_initiate_chat(recipient, problem=contents)
    else:
        await init_sender.a_initiate_chat(recipient, message=contents)


async def reply_chat(contents, user, instance):
    if hasattr(instance, "collection_name"):
        collection_name = instance.collection_name
    else:
        collection_name = f"{int(time.time())}_{random.randint(0, 100000)}"
        instance.collection_name = collection_name

    column_agents_list = [[a.value for a in agent[0]] for agent in column_agents] + [switch_code.value]
    if not hasattr(instance, "agent_list") or instance.agents_list != column_agents_list:
        agents, manager, groupchat = init_groupchat(None, collection_name)
        instance.manager = manager
        instance.agents = agents
        instance.agents_list = column_agents_list
    else:
        agents = instance.agents
        manager = instance.manager

    if len(agents) <= 1:
        return "Please add more agents."

    init_sender = None
    for agent in agents:
        if "UserProxy" in str(type(agent)):
            init_sender = agent
            break
    for agent in agents:
        # Hack for get human input
        agent._reply_func_list.pop(1)
        agent.register_reply(
            [autogen.Agent, None],
            partial(check_termination_and_human_reply, instance=instance),
            1,
        )
    if manager is not None:
        for agent in agents:
            agent._reply_func_list.pop(0)

    if not init_sender:
        init_sender = agents[0]
    await generate_code(agents, manager, contents, code_editor, groupchat)
    await agents_chat(init_sender, manager, contents, agents)
    return "The task is done. Please start a new task."


chatiface = ChatInterface(
    callback=reply_chat,
    height=600,
)

template.main.append(chatiface)

btn_msg1 = Button(name=Q1, sizing_mode="stretch_width")
btn_msg2 = Button(name=Q2, sizing_mode="stretch_width")
btn_msg3 = Button(name=Q3, sizing_mode="stretch_width")
template.main.append(
    pn.Column(
        pn.pane.Markdown("## Message Examples: ", sizing_mode="stretch_width"),
        btn_msg1,
        btn_msg2,
        btn_msg3,
        sizing_mode="stretch_width",
    )
)


def load_message(event):
    if event.obj.name == Q1:
        chatiface.send(Q1)
    elif event.obj.name == Q2:
        chatiface.send(Q2)
    elif event.obj.name == Q3:
        chatiface.send(Q3)


btn_msg1.on_click(load_message)
btn_msg2.on_click(load_message)
btn_msg3.on_click(load_message)


btn_example1 = Button(name="General 2 agents", button_type="primary", sizing_mode="stretch_width")
btn_example2 = Button(name="RAG 2 agents", button_type="primary", sizing_mode="stretch_width")
btn_example3 = Button(name="Software Dev 3 agents", button_type="primary", sizing_mode="stretch_width")
btn_example4 = Button(name="Research 6 agents", button_type="primary", sizing_mode="stretch_width")
btn_example5 = Button(name="Debate 9 agents", button_type="primary", sizing_mode="stretch_width")
template.main.append(
    pn.Row(
        pn.pane.Markdown("## Agent Examples: ", sizing_mode="stretch_width"),
        btn_example1,
        btn_example2,
        btn_example3,
        btn_example4,
        btn_example5,
        sizing_mode="stretch_width",
    )
)


def clear_agents():
    while len(column_agents) > 0:
        column_agents.pop(-1)


def load_example(event):
    clear_agents()
    if event.obj.name == "RAG 2 agents":
        column_agents.append(
            RowAgentWidget(
                value=[
                    "Boss_Assistant",
                    "Assistant who has extra content retrieval power for solving difficult problems.",
                    "RetrieveUserProxyAgent",
                    "",
                ]
            ),
        )
        column_agents.append(
            RowAgentWidget(
                value=[
                    "Senior_Python_Engineer",
                    f"You are a senior python engineer. {DEFAULT_TERMINATE_MESSAGE}",
                    "RetrieveAssistantAgent",
                    "",
                ]
            ),
        )
    elif event.obj.name == "General 2 agents":
        column_agents.append(
            RowAgentWidget(
                value=[
                    "User_Proxy",
                    "",
                    "UserProxyAgent",
                    "",
                ]
            ),
        )
        column_agents.append(
            RowAgentWidget(
                value=[
                    "Assistant_Agent",
                    "",
                    "AssistantAgent",
                    "",
                ]
            ),
        )
    elif event.obj.name == "Software Dev 3 agents":
        column_agents.append(
            RowAgentWidget(
                value=[
                    "Boss",
                    f"The boss who ask questions and give tasks. {DEFAULT_TERMINATE_MESSAGE}",
                    "UserProxyAgent",
                    "",
                ]
            ),
        )
        column_agents.append(
            RowAgentWidget(
                value=[
                    "Senior_Python_Engineer",
                    f"You are a senior python engineer. {DEFAULT_TERMINATE_MESSAGE}",
                    "AssistantAgent",
                    "",
                ]
            ),
        )
        column_agents.append(
            RowAgentWidget(
                value=[
                    "Product_Manager",
                    f"You are a product manager. {DEFAULT_TERMINATE_MESSAGE}",
                    "AssistantAgent",
                    "",
                ]
            ),
        )
    elif event.obj.name == "Research 6 agents":
        column_agents.append(
            RowAgentWidget(
                value=[
                    "Admin",
                    "A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin.",
                    "UserProxyAgent",
                    "",
                ]
            ),
        )
        column_agents.append(
            RowAgentWidget(
                value=[
                    "Engineer",
                    """Engineer. You follow an approved plan. You write python/shell code to solve tasks. Wrap the code in a code block that specifies the script type. The user can't modify your code. So do not suggest incomplete code which requires others to modify. Don't use a code block if it's not intended to be executed by the executor.
Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. Check the execution result returned by the executor.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
""",
                    "AssistantAgent",
                    "",
                ]
            ),
        )
        column_agents.append(
            RowAgentWidget(
                value=[
                    "Scientist",
                    """Scientist. You follow an approved plan. You are able to categorize papers after seeing their abstracts printed. You don't write code.""",
                    "AssistantAgent",
                    "",
                ]
            ),
        )
        column_agents.append(
            RowAgentWidget(
                value=[
                    "Planner",
                    """Planner. Suggest a plan. Revise the plan based on feedback from admin and critic, until admin approval.
The plan may involve an engineer who can write code and a scientist who doesn't write code.
Explain the plan first. Be clear which step is performed by an engineer, and which step is performed by a scientist.
""",
                    "AssistantAgent",
                    "",
                ]
            ),
        )
        column_agents.append(
            RowAgentWidget(
                value=[
                    "Critic",
                    "Critic. Double check plan, claims, code from other agents and provide feedback. Check whether the plan includes adding verifiable info such as source URL.",
                    "AssistantAgent",
                    "",
                ]
            ),
        )

        column_agents.append(
            RowAgentWidget(
                value=[
                    "Executor",
                    "Executor. Execute the code written by the engineer and report the result.",
                    "UserProxyAgent",
                    "",
                ]
            ),
        )
    elif event.obj.name == "Debate 9 agents":
        column_agents.append(
            RowAgentWidget(
                value=[
                    "Host",
                    "辩论主持人。主持辩论，提出问题，总结辩论结果",
                    "AssistantAgent",
                    "",
                ]
            ),
        )
        column_agents.append(
            RowAgentWidget(
                value=[
                    "team_one_member_one",
                    "你是辩论队的正方一辩，你要根据给定的题目说出正方观点。你总是以`我是正方一辩...`开头。",
                    "AssistantAgent",
                    "",
                ]
            ),
        )
        column_agents.append(
            RowAgentWidget(
                value=[
                    "team_two_member_one",
                    "你是辩论队的反方一辩，你要根据给定的题目说出反方观点。你可以在辩论过程中反驳正方一辩的观点。你总是以`我是反方一辩...`开头。",
                    "AssistantAgent",
                    "",
                ]
            ),
        )
        column_agents.append(
            RowAgentWidget(
                value=[
                    "team_one_member_two",
                    "你是辩论队的正方二辩，你要根据给定的题目说出正方观点。你可以在辩论过程中反驳反方一辩的观点。你总是以`我是正方二辩...`开头。",
                    "AssistantAgent",
                    "",
                ]
            ),
        )
        column_agents.append(
            RowAgentWidget(
                value=[
                    "team_two_member_two",
                    "你是辩论队的反方二辩，你要根据给定的题目说出反方观点。你可以在辩论过程中反驳正方二辩的观点。你总是以`我是反方二辩...`开头。",
                    "AssistantAgent",
                    "",
                ]
            ),
        )
        column_agents.append(
            RowAgentWidget(
                value=[
                    "team_one_member_three",
                    "你是辩论队的正方三辩，你要根据给定的题目说出正方观点。你可以在辩论过程中反驳反方二辩的观点。你总是以`我是正方三辩...`开头。",
                    "AssistantAgent",
                    "",
                ]
            ),
        )
        column_agents.append(
            RowAgentWidget(
                value=[
                    "team_two_member_there",
                    "你是辩论队的反方三辩，你要根据给定的题目说出反方观点。你可以在辩论过程中反驳正方三辩的观点。你总是以`我是反方三辩...`开头。",
                    "AssistantAgent",
                    "",
                ]
            ),
        )
        column_agents.append(
            RowAgentWidget(
                value=[
                    "team_one_member_four",
                    "你是辩论队的正方四辩，你要总结前面的辩论内容，言简意赅的再次反驳反方观点，并总结说明正方观点。你总是以`我是正方四辩...`开头。",
                    "AssistantAgent",
                    "",
                ]
            ),
        )
        column_agents.append(
            RowAgentWidget(
                value=[
                    "team_two_member_four",
                    "你是辩论队的反方四辩，你要总结前面的辩论内容，言简意赅的再次反驳正方观点，并总结说明反方观点。你总是以`我是反方四辩...`开头。",
                    "AssistantAgent",
                    "",
                ]
            ),
        )


btn_example1.on_click(load_example)
btn_example2.on_click(load_example)
btn_example3.on_click(load_example)
btn_example4.on_click(load_example)
btn_example5.on_click(load_example)

code_editor = CodeEditor(value="", sizing_mode="stretch_width", language="python", height=300)
template.main.append(code_editor)

template.servable(title=TITLE)
