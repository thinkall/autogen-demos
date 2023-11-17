import atexit
import os
import sys
import threading
from functools import partial
from itertools import chain
from typing import Union

import anyio
import autogen
import gradio as gr
from autogen import Agent, AssistantAgent, OpenAIWrapper, UserProxyAgent
from autogen.code_utils import extract_code
from diskcache import Cache
from gradio import ChatInterface, Request
from gradio.helpers import special_args
from pydantic.dataclasses import dataclass

LOG_LEVEL = "INFO"
TIMEOUT = 10  # seconds
CACHE_EXPIRE_TIME = 7200  # 2 hours
_cache = Cache(".cache/gradio")


def close_cache():
    _cache.close()


atexit.register(close_cache)


class myChatInterface(ChatInterface):
    async def _submit_fn(
        self,
        message: str,
        history_with_input: list[list[str | None]],
        request: Request,
        *args,
    ) -> tuple[list[list[str | None]], list[list[str | None]]]:
        history = history_with_input[:-1]
        inputs, _, _ = special_args(self.fn, inputs=[message, history, *args], request=request)

        if self.is_async:
            await self.fn(*inputs)
        else:
            await anyio.to_thread.run_sync(self.fn, *inputs, limiter=self.limiter)

        # history.append([message, response])
        return history, history


@dataclass
class AgentHistory:
    session_hash: str = None
    sender: str = None
    messages: list = None


def get_history(session_hash):
    return _cache.get(session_hash, [])


def save_history(session_hash, history: AgentHistory):
    _cache.set(session_hash, get_history(session_hash).append(history), expire=CACHE_EXPIRE_TIME)


def delete_history(session_hash):
    _cache.delete(session_hash)
