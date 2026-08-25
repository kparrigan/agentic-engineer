from typing import Annotated, TypedDict
from langgraph.graph import add_messages

class AgentState(TypedDict):
    """Represents the state of an agent in the system."""
    messages: Annotated[list, add_messages] #short term memory. Merge instead of add
    goal: str
    iterations: int
    final_answer: str|None