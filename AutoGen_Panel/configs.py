import autogen

TIMEOUT = 60
TITLE = "Microsoft AutoGen Playground"
Q1 = "What's AutoGen?"
Q2 = "Write a python function to compute the sum of numbers."
Q3 = "find papers on LLM applications from arxiv in the last week, create a markdown table of different domains."

DEFAULT_SYSTEM_MESSAGE = autogen.AssistantAgent.DEFAULT_SYSTEM_MESSAGE
DEFAULT_AUTO_REPLY = """Reply "TERMINATE" in the end when everything is done."""
