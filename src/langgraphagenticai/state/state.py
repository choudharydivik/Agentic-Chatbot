from typing import TypedDict, List
from langchain_core.messages import BaseMessage
from typing_extensions import Annotated
from langgraph.graph.message import add_messages

class State(TypedDict, total=False):
    # LangGraph required
    messages: Annotated[List[BaseMessage], add_messages]

    # AI News specific
    frequency: str
    news_data: list
    summary: str
    filename: str
