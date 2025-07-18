from langchain_core.messages import ToolMessage
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