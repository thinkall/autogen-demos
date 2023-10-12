import gradio as gr
import os
import shutil
import autogen
import chromadb
import multiprocessing as mp
from autogen.oai.openai_utils import config_list_from_json
from autogen.retrieve_utils import TEXT_FORMATS
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent, PROMPT_DEFAULT


def setup_configurations(config_file="OAI_CONFIG_LIST"):
    config_list = autogen.config_list_from_json(
        env_or_file=config_file,
        file_location=".",
        filter_dict={
            "model": {
                "gpt-4",
                "gpt4",
                "gpt-4-32k",
                "gpt-4-32k-0314",
                "gpt-35-turbo",
                "gpt-3.5-turbo",
            }
        },
    )
    assert len(config_list) > 0
    print("models to use: ", [config_list[i]["model"] for i in range(len(config_list))])

    return config_list


def initialize_agents(config_list, docs_path=None):
    if docs_path is None:
        docs_path = "https://raw.githubusercontent.com/microsoft/autogen/main/README.md"
    autogen.ChatCompletion.start_logging()

    assistant = RetrieveAssistantAgent(
        name="assistant",
        system_message="You are a helpful assistant.",
        llm_config={
            "request_timeout": 600,
            "seed": 42,
            "config_list": config_list,
        },
    )

    ragproxyagent = RetrieveUserProxyAgent(
        name="ragproxyagent",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=5,
        retrieve_config={
            # "task": "qa",
            "docs_path": docs_path,
            "chunk_token_size": 2000,
            "model": config_list[0]["model"],
            "client": chromadb.PersistentClient(path="/tmp/chromadb1"),
            "embedding_model": "all-mpnet-base-v2",
            "customized_prompt": PROMPT_DEFAULT,
        },
    )

    return assistant, ragproxyagent


def initiate_chat(problem, queue, n_results=3):
    global assistant, ragproxyagent
    if assistant is None:
        queue.put(["Please upload the LLM config file first"])
        return
    assistant.reset()
    ragproxyagent.initiate_chat(assistant, problem=problem, silent=False, n_results=n_results)
    # queue.put(ragproxyagent.last_message()["content"])
    messages = ragproxyagent.chat_messages
    messages = [messages[k] for k in messages.keys()][0]
    messages = [m["content"] for m in messages if m["role"] == "user"]
    print("messages: ", messages)
    queue.put(messages)


def chatbot_reply(input_text):
    """Chat with the agent through terminal."""
    queue = mp.Queue()
    process = mp.Process(
        target=initiate_chat,
        args=(input_text, queue),
    )
    process.start()
    process.join()
    messages = queue.get()
    return messages

def get_description_text():
    return """
    # Microsoft AutoGen: Retrieve Chat Demo
    
    This demo shows how to use the RetrieveUserProxyAgent and RetrieveAssistantAgent to build a chatbot.

    #### [GitHub](https://github.com/microsoft/autogen) [Discord](https://discord.gg/pAbnFJrkgZ) [Docs](https://microsoft.github.io/autogen/) [Paper](https://arxiv.org/abs/2308.08155)

    LLM configure file should contain OpenAI, Azure OpenAI or other openai compatible models, for example:
    ```
        [
            {
                "engine": "gpt-35-turbo", 
                "model": "gpt-3.5-turbo",
                "api_base": "https://xxx.openai.azure.com", 
                "api_type": "azure", 
                "api_version": "2023-05-15", 
                "api_key": "xxx",
            }
        ]
    ```
    """

global config_list, assistant, ragproxyagent
assistant = None
with gr.Blocks() as demo:
    gr.Markdown(get_description_text())
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=(None, (os.path.join(os.path.dirname(__file__), "autogen.png"))),
        # height=600,
    )
    with gr.Row():
        txt_input = gr.Textbox(
            scale=4,
            show_label=False,
            placeholder="Enter text and press enter",
            container=False,
        )

        def upload_file(file):
            global config_list, assistant, ragproxyagent
            config_list = setup_configurations(config_file=file.name)
            assistant, ragproxyagent = initialize_agents(config_list)

        upload_button = gr.UploadButton("Click to Upload LLM Config File", file_types=["file"], file_count="single")
        upload_button.upload(upload_file, upload_button)

    clear = gr.ClearButton([txt_input, chatbot])

    txt_context_url = gr.Textbox(
        label="Enter the url to your context file and chat on the context",
        info=f"File must be in the format of [{', '.join(TEXT_FORMATS)}]",
        max_lines=1,
        show_label=True,
        value="https://raw.githubusercontent.com/microsoft/autogen/main/README.md",
        container=True,
    )

    txt_prompt = gr.Textbox(
        label="Enter your prompt for Retrieve Agent and press enter to replace the default prompt",
        max_lines=40,
        show_label=True,
        value=PROMPT_DEFAULT,
        container=True,
        show_copy_button=True,
        layout={"height": 20},
    )


    def respond(message, chat_history):
        messages = chatbot_reply(message)
        chat_history.append((message, messages[-1] if messages[-1] != "TERMINATE" else messages[-2]))
        return "", chat_history

    def update_prompt(prompt):
        ragproxyagent.customized_prompt = prompt
        return prompt
    
    def update_context_url(context_url):
        global assistant, ragproxyagent
        try:
            shutil.rmtree("/tmp/chromadb1/")
        except:
            pass
        assistant, ragproxyagent = initialize_agents(config_list, docs_path=context_url)
        return context_url
    
    txt_input.submit(respond, [txt_input, chatbot], [txt_input, chatbot])
    txt_prompt.submit(update_prompt, [txt_prompt], [txt_prompt])
    txt_context_url.submit(update_context_url, [txt_context_url], [txt_context_url])


if __name__ == "__main__":
    demo.launch(share=True)

