import os
import time
from functools import partial

import autogen
import gradio as gr
from autogen import Agent, AssistantAgent, OpenAIWrapper, UserProxyAgent
from autogen_utils import (
    LOG_LEVEL,
    TIMEOUT,
    chat_to_oai_message,
    get_history,
    initialize_agents,
    myChatInterface,
    oai_message_to_chat,
    thread_with_trace,
    update_agent_history,
)
from gradio import Request


def initiate_chat(config_list, user_message, chat_history, session_hash):
    if LOG_LEVEL == "DEBUG":
        print(f"chat_history_init: {chat_history}")
    # agent_history = flatten_chain(chat_history)
    if len(config_list[0].get("api_key", "")) < 2:
        chat_history.append(
            [
                user_message,
                "Hi, nice to meet you! Please enter your API keys in below text boxs.",
            ]
        )
        return chat_history
    else:
        llm_config = {
            # "seed": 42,
            "timeout": TIMEOUT,
            "config_list": config_list,
        }
        assistant.llm_config.update(llm_config)
        assistant.client = OpenAIWrapper(**assistant.llm_config)

    if user_message.strip().lower().startswith("show file:"):
        filename = user_message.strip().lower().replace("show file:", "").strip()
        filepath = os.path.join("coding", filename)
        if os.path.exists(filepath):
            chat_history.append([user_message, (filepath,)])
        else:
            chat_history.append([user_message, f"File {filename} not found."])
        return chat_history

    assistant.reset()
    oai_messages = chat_to_oai_message(chat_history)
    assistant._oai_system_message_origin = assistant._oai_system_message.copy()
    assistant._oai_system_message += oai_messages

    assistant.register_reply([Agent, None], partial(update_agent_history, session_hash=session_hash))
    userproxy.register_reply([Agent, None], partial(update_agent_history, session_hash=session_hash))

    try:
        userproxy.initiate_chat(assistant, message=user_message)
        messages = userproxy.chat_messages
        chat_history += oai_message_to_chat(messages, assistant)
        # agent_history = flatten_chain(chat_history)
    except Exception as e:
        # agent_history += [user_message, str(e)]
        # chat_history[:] = agent_history_to_chat(agent_history)
        chat_history.append([user_message, str(e)])

    assistant._oai_system_message = assistant._oai_system_message_origin.copy()
    if LOG_LEVEL == "DEBUG":
        print(f"chat_history: {chat_history}")
        # print(f"agent_history: {agent_history}")
    return chat_history


def chatbot_reply_thread(input_text, chat_history, config_list, session_hash):
    """Chat with the agent through terminal."""
    thread = thread_with_trace(target=initiate_chat, args=(config_list, input_text, chat_history, session_hash))
    thread.start()
    try:
        messages = thread.join(timeout=TIMEOUT)
        if thread.is_alive():
            thread.kill()
            thread.join()
            messages = [
                input_text,
                "Timeout Error: Please check your API keys and try again later.",
            ]
    except Exception as e:
        messages = [
            [
                input_text,
                str(e) if len(str(e)) > 0 else "Invalid Request to OpenAI, please check your API keys.",
            ]
        ]
    return messages


def chatbot_reply_plain(input_text, chat_history, config_list, session_hash):
    """Chat with the agent through terminal."""
    try:
        messages = initiate_chat(config_list, input_text, chat_history, session_hash)
    except Exception as e:
        messages = [
            [
                input_text,
                str(e) if len(str(e)) > 0 else "Invalid Request to OpenAI, please check your API keys.",
            ]
        ]
    return messages


with gr.Blocks() as demo:
    config_list, assistant, userproxy = (
        [
            {
                "api_key": "",
                "base_url": "",
                "api_type": "azure",
                "api_version": "2023-07-01-preview",
                "model": "gpt-35-turbo",
            }
        ],
        None,
        None,
    )
    assistant, userproxy = initialize_agents(config_list)

    def get_description_text():
        return """
        # Microsoft AutoGen: Multi-Round Human Interaction Chatbot Demo

        This demo shows how to build a chatbot which can handle multi-round conversations with human interactions.

        #### [[AutoGen](https://github.com/microsoft/autogen)] [[Discord](https://discord.gg/pAbnFJrkgZ)] [[Paper](https://arxiv.org/abs/2308.08155)] [[SourceCode](https://github.com/thinkall/autogen-demos)]
        """

    description = gr.Markdown(get_description_text())

    def update_config():
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

        return config_list

    def set_params(model, oai_key, aoai_key, aoai_base):
        os.environ["MODEL"] = model
        os.environ["OPENAI_API_KEY"] = oai_key
        os.environ["AZURE_OPENAI_API_KEY"] = aoai_key
        os.environ["AZURE_OPENAI_API_BASE"] = aoai_base

    def chatbot_reply(input_text, chat_history, config_list, session_hash):
        """Chat with the agent through terminal."""
        return chatbot_reply_thread(input_text, chat_history, config_list, session_hash)

    def update_chatbot(session_hash, chat_history):
        print(f"{id(chat_history)=}")
        while True:
            cache_history = get_history(session_hash)
            cache_history = [] if not cache_history else cache_history
            print(f"cache_history: {cache_history}")
            chat_history[:] = [msg.message for msg in cache_history]
            time.sleep(0.1)

    def respond(message, chat_history, model, oai_key, aoai_key, aoai_base, request: Request):
        if request:
            session_hash = dict(request.query_params).get("session_hash")
        else:
            return ""
        print(f"session_hash: {session_hash}, type: {type(session_hash)}")
        set_params(model, oai_key, aoai_key, aoai_base)
        config_list = update_config()
        print(f"{id(chat_history)=}")
        thread = thread_with_trace(target=update_chatbot, args=(session_hash, chat_history))
        thread.start()

        chatbot_reply(message, chat_history, config_list, session_hash)

        thread.join()
        if thread.is_alive():
            thread.kill()
            thread.join()

        return ""

    with gr.Row() as params:
        txt_model = gr.Dropdown(
            label="Model",
            choices=[
                "gpt-4",
                "gpt-35-turbo",
                "gpt-3.5-turbo",
            ],
            allow_custom_value=True,
            value="gpt-35-turbo",
            container=True,
        )
        txt_oai_key = gr.Textbox(
            label="OpenAI API Key",
            placeholder="Enter OpenAI API Key",
            max_lines=1,
            show_label=True,
            container=True,
            type="password",
        )
        txt_aoai_key = gr.Textbox(
            label="Azure OpenAI API Key",
            placeholder="Enter Azure OpenAI API Key",
            max_lines=1,
            show_label=True,
            container=True,
            type="password",
        )
        txt_aoai_base_url = gr.Textbox(
            label="Azure OpenAI API Base",
            placeholder="Enter Azure OpenAI Base Url",
            max_lines=1,
            show_label=True,
            container=True,
            type="password",
        )

    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=(
            "autogen-icons/user_green.png",
            (os.path.join(os.path.dirname(__file__), "autogen-icons", "agent_blue.png")),
        ),
        render=False,
        height=600,
    )

    txt_input = gr.Textbox(
        scale=4,
        show_label=False,
        placeholder="Enter text and press enter",
        container=False,
        render=False,
        autofocus=True,
    )

    chatiface = myChatInterface(
        respond,
        chatbot=chatbot,
        textbox=txt_input,
        additional_inputs=[
            txt_model,
            txt_oai_key,
            txt_aoai_key,
            txt_aoai_base_url,
        ],
        examples=[
            ["write a python function to count the sum of two numbers?"],
            ["what if the production of two numbers?"],
            [
                "Plot a chart of the last year's stock prices of Microsoft, Google and Apple and save to stock_price.png."
            ],
            ["show file: stock_price.png"],
        ],
    )


if __name__ == "__main__":
    demo.queue().launch(share=True, server_name="0.0.0.0")
