import gradio as gr
import os
import threading
import sys
import autogen
from autogen.code_utils import extract_code
from autogen import UserProxyAgent, AssistantAgent, Agent, OpenAIWrapper


TIMEOUT = 60


class thread_with_trace(threading.Thread):
    # https://www.geeksforgeeks.org/python-different-ways-to-kill-a-thread/
    # https://stackoverflow.com/questions/6893968/how-to-get-the-return-value-from-a-thread
    def __init__(self, *args, **keywords):
        threading.Thread.__init__(self, *args, **keywords)
        self.killed = False
        self._return = None

    def start(self):
        self.__run_backup = self.run
        self.run = self.__run
        threading.Thread.start(self)

    def __run(self):
        sys.settrace(self.globaltrace)
        self.__run_backup()
        self.run = self.__run_backup

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def globaltrace(self, frame, event, arg):
        if event == "call":
            return self.localtrace
        else:
            return None

    def localtrace(self, frame, event, arg):
        if self.killed:
            if event == "line":
                raise SystemExit()
        return self.localtrace

    def kill(self):
        self.killed = True

    def join(self, timeout=0):
        threading.Thread.join(self, timeout)
        return self._return


def update_chat_history(recipient, messages, sender, config):
    if config is None:
        config = recipient
    if messages is None:
        messages = recipient._oai_messages[sender]
    message = messages[-1]
    print(f"Messages sent to: {recipient.name} | num messages: {len(messages)}")
    return False, None  # required to ensure the agent communication flow continues


def _is_termination_msg(message):
    """Check if a message is a termination message.
    Terminate when no code block is detected. Currently only detect python code blocks.
    """
    if isinstance(message, dict):
        message = message.get("content")
        if message is None:
            return False
    cb = extract_code(message)
    contain_code = False
    for c in cb:
        # todo: support more languages
        if c[0] == "python":
            contain_code = True
            break
    return not contain_code


def initialize_agents(config_list):
    assistant = AssistantAgent(
        name="assistant",
        max_consecutive_auto_reply=5,
        llm_config={
            "seed": 42,
            "timeout": TIMEOUT,
            "config_list": config_list,
        },
    )

    userproxy = UserProxyAgent(
        name="userproxy",
        human_input_mode="NEVER",
        is_termination_msg=_is_termination_msg,
        max_consecutive_auto_reply=5,
        # code_execution_config=False,
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False,  # set to True or image name like "python:3" to use docker
        },
    )

    # assistant.register_reply([Agent, None], update_chat_history)
    # userproxy.register_reply([Agent, None], update_chat_history)

    return assistant, userproxy


def chat_to_oai_message(chat_history):
    """Convert chat history to OpenAI message format."""
    messages = []
    for msg in chat_history:
        messages.append({"content": msg[0], "role": "user"})
        messages.append({"content": msg[1], "role": "assistant"})
    return messages


def oai_message_to_chat(oai_messages, sender):
    """Convert OpenAI message format to chat history."""
    chat_history = []
    messages = oai_messages[sender]
    for i in range(len(messages) // 2):
        chat_history.append(
            [messages[2 * i]["content"], messages[2 * i + 1]["content"]]
        )
    return chat_history


def initiate_chat(config_list, user_message, chat_history):
    global assistant, userproxy
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
            "seed": 42,
            "timeout": TIMEOUT,
            "config_list": config_list,
        }
        assistant.llm_config.update(llm_config)
        assistant.client = OpenAIWrapper(**assistant.llm_config)

    assistant.reset()
    oai_messages = chat_to_oai_message(chat_history)
    assistant._oai_system_message += oai_messages
    userproxy.initiate_chat(assistant, message=user_message)
    try:
        messages = userproxy.chat_messages
        chat_history += oai_message_to_chat(messages, assistant)
    except Exception as e:
        chat_history += [[user_message, str(e)]]
    return chat_history


def chatbot_reply(input_text, chat_history, config_list):
    """Chat with the agent through terminal."""
    thread = thread_with_trace(
        target=initiate_chat, args=(config_list, input_text, chat_history)
    )
    thread.start()
    try:
        messages = thread.join(timeout=TIMEOUT)
        if thread.is_alive():
            thread.kill()
            thread.join()
            messages = [
                "Timeout Error: Please check your API keys and try again later."
            ]
    except Exception as e:
        messages = [
            str(e)
            if len(str(e)) > 0
            else "Invalid Request to OpenAI, please check your API keys."
        ]
    return messages


def get_description_text():
    return """
    # Microsoft AutoGen: Retrieve Chat Demo
    
    This demo shows how to use the RetrieveUserProxyAgent and RetrieveAssistantAgent to build a chatbot.

    #### [AutoGen](https://github.com/microsoft/autogen) [Discord](https://discord.gg/pAbnFJrkgZ) [Blog](https://microsoft.github.io/autogen/blog/2023/10/18/RetrieveChat) [Paper](https://arxiv.org/abs/2308.08155) [SourceCode](https://github.com/thinkall/autogen-demos)
    """


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

    description = gr.Markdown(get_description_text())
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=(None, (os.path.join(os.path.dirname(__file__), "autogen.png"))),
    )

    txt_input = gr.Textbox(
        scale=4,
        show_label=False,
        placeholder="Enter text and press enter",
        container=False,
    )

    with gr.Row() as params:

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
            return model, oai_key, aoai_key, aoai_base

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
            placeholder="Enter key and press enter",
            max_lines=1,
            show_label=True,
            container=True,
            type="password",
        )
        txt_aoai_key = gr.Textbox(
            label="Azure OpenAI API Key",
            placeholder="Enter key and press enter",
            max_lines=1,
            show_label=True,
            container=True,
            type="password",
        )
        txt_aoai_base_url = gr.Textbox(
            label="Azure OpenAI API Base",
            placeholder="Enter base url and press enter",
            max_lines=1,
            show_label=True,
            container=True,
            type="password",
        )

    clear = gr.ClearButton([txt_input, chatbot])

    def respond(message, chat_history, model, oai_key, aoai_key, aoai_base):
        global config_list
        set_params(model, oai_key, aoai_key, aoai_base)
        config_list = update_config()
        chat_history = chatbot_reply(message, chat_history, config_list)
        return "", chat_history

    txt_input.submit(
        respond,
        [txt_input, chatbot, txt_model, txt_oai_key, txt_aoai_key, txt_aoai_base_url],
        [txt_input, chatbot],
    )

    def print_chat_history(chat_history):
        print(f"Chat History Length: {len(chat_history)}")

    # demo.load(print_chat_history, chatbot, None, every=1)


if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0")
