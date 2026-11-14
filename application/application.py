import asyncio

from dotenv import load_dotenv
load_dotenv()
import immich_tools
from graph import build_graph
from langchain_core.messages import HumanMessage

async def main():
    graph = build_graph()
    config = {"configurable": {"thread_id": "thread_name"}}
    input = [HumanMessage(content="Tell me how are you!")]
    output = graph.invoke({"messages" : input}, config)

    for message in output["messages"]:
        message.pretty_print()


if __name__ == "__main__":
    asyncio.run(main())
