import panel as pn
import param
from panel.viewable import Viewer
from panel.widgets import Button, PasswordInput, Select, TextAreaInput, TextInput

DEFAULT_LIST = [
    "https://raw.githubusercontent.com/microsoft/autogen/main/README.md",
    "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Examples/Integrate%20-%20Spark.md",
]


class RowAgentWidget(Viewer):
    """A widget for creating a row of agent widgets. Including agent name, system message, and agent type."""

    value = param.List(
        default=[
            "",
            "",
            "AssistantAgent",
            DEFAULT_LIST,
        ],
        doc="Agent name, system message, and agent type",
    )

    def __init__(self, **params):
        layout_params = {key: value for key, value in params.items() if key not in ["value"]}
        params = {key: value for key, value in params.items() if key not in layout_params}
        self._agent_name = TextInput(
            name="",
            value=params.get("value")[0],
            placeholder="Agent Name",
            min_width=100,
            sizing_mode="scale_width",
        )
        self._system_msg = TextInput(
            name="",
            value=params.get("value")[1],
            placeholder="System Message, leave empty to use default",
            min_width=400,
            sizing_mode="scale_width",
        )
        self._agent_type = Select(
            name="",
            value=params.get("value")[2],
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
                # "TeachableAgent",
            ],
            sizing_mode="scale_width",
        )
        self._rag_docs = TextAreaInput(
            name="",
            value=f"{DEFAULT_LIST}",
            placeholder="List of links to docs",
            sizing_mode="scale_width",
            auto_grow=True,
            visible=False,
        )

        super().__init__(**params)
        self._layout = pn.Column(
            pn.Row(self._agent_name, self._system_msg, self._agent_type, sizing_mode="scale_width"),
            self._rag_docs,
            sizing_mode="scale_width",
        )
        self._sync_widgets()

    def __panel__(self):
        return self._layout

    @param.depends("value", watch=True)
    def _sync_widgets(self):
        self._agent_name.value = self.value[0]
        self._system_msg.value = self.value[1]
        self._agent_type.value = self.value[2]
        if self.value[2] == "RetrieveUserProxyAgent":
            self._rag_docs.visible = True
        else:
            self._rag_docs.visible = False

    @param.depends("_agent_name.value", "_system_msg.value", "_agent_type.value", "_rag_docs.value", watch=True)
    def _sync_params(self):
        self.value = [self._agent_name.value, self._system_msg.value, self._agent_type.value, self._rag_docs.value]
