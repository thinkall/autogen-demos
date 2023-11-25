import autogen

TIMEOUT = 60
TITLE = "Microsoft AutoGen Playground"
Q1 = "What's AutoGen?"
Q2 = "Write a python function to compute the sum of numbers."
Q3 = "find papers on LLM applications from arxiv in the last week, create a markdown table of different domains."

DEFAULT_SYSTEM_MESSAGE = autogen.AssistantAgent.DEFAULT_SYSTEM_MESSAGE
DEFAULT_TERMINATE_MESSAGE = "Reply `TERMINATE` in the end if the task is done."
DEFAULT_AUTO_REPLY = "Thank you. Reply `TERMINATE` to exit."
