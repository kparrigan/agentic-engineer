from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI    # swap for your provider

from agent.state import AgentState
from agent.tools import TOOLS
from agent.prompt import SYSTEM_PROMPT

def build_graph(config):
    """Factory: build and compile the agent graph from a config.
    Chapter 3's harness calls build_graph(config) and wraps the result.
    Keeping construction in a factory (not at module scope) is what lets the
    next chapter bind a configured model and tool set without editing this file."""

    model_name = getattr(config, "model_name", "gpt-4o")
    temperature = getattr(config, "temperature", 0.0)
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    llm_with_tools = llm.bind_tools(list(TOOLS.values()))

    def reason_node(state: AgentState)-> dict:
        """PERCEIVE + REASON: read state, call the tool-bound model, return a
            partial-state UPDATE. We never mutate `state` in place — LangGraph merges
            the returned delta via the reducers (add_messages handles `messages`)."""

        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        update = {
            "messages": [response],                 # merged by add_messages reducer
            "iterations": state["iterations"] + 1,
        }

        if not response.tool_calls:
            update["final_answer"] = response.content

        return update

    def act_node(state: AgentState)-> dict:
        """ACT + OBSERVE: execute the requested tool(s), return the results as a
        partial-state UPDATE. Again: return the delta, never mutate `state`."""
        last = state["messages"][-1]
        observations = []

        for call in last.tool_calls:
            fn = TOOLS[call["name"]]
            result = fn(**call["args"])
            observations.append(
                ToolMessage(tool_call_id=call["id"], content=str(result))
            )

        return {"messages": observations}

    def route(state: AgentState)->str:
        """Did the model ask for a tool, or give a final answer?
        NOTE: this is the SOFT, model-driven stop only. Real exit conditions —
        budgets, oscillation guards, acceptance checks — are Chapter 3's job.
        This single check is intentionally not enough to ship; Chapter 3 replaces
        it with a full `evaluate_exit`."""        

        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "act"

        return "end"

    graph = StateGraph(AgentState)
    graph.add_node("reason", reason_node)
    graph.add_node("act", act_node)
    graph.set_entry_point("reason")
    graph.add_conditional_edges("reason", route, {"act": "act", "end": END})
    graph.add_edge("act", "reason")
    
    return graph.compile()
