from langchain_core.messages import ToolMessage
from langgraph.graph import END
from langchain_core.runnables import RunnableConfig

from literature_review_agent.state import LitState

from typing import Optional
import json


class AsyncToolNode:
    def __init__(self, tools):
        self.tools = {t.name: t for t in tools}

    async def __call__(self, inputs):
        msgs = inputs["messages"]
        last = msgs[-1]
        out = []
        for call in last.tool_calls:
            tool_name = call["name"]
            tool_args = call["args"]
            print(f"Invoking tool: {tool_name} with args: {tool_args}\n")
            result = await self.tools[tool_name].ainvoke(tool_args)
            out.append(
                ToolMessage(
                    name=call["name"],
                    tool_call_id=call["id"],
                    content=json.dumps(result),
                )
            )
        return {"messages": msgs + out}
    

def route_tools(state: LitState, *, config: Optional[RunnableConfig] = None) -> dict:
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """

    print("Checking for tool calls...\n")

    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


def set_workflow_completed_flag(state: LitState, *, config: Optional[RunnableConfig] = None) -> dict:
    return {
        "completed": True
    }