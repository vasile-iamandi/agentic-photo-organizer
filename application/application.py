import asyncio

from dotenv import load_dotenv
load_dotenv()
import immich_tools
from graph import build_graph
from langchain_core.messages import HumanMessage

async def main():
    graph = build_graph()
    config = {"configurable": {"thread_id": "thread_name"}}
    input = [HumanMessage(content="Arata-mi niste poze cu pisici")]
    output = await graph.ainvoke({"messages" : input}, config)

    for message in output["messages"]:
        message.pretty_print()

#    assets_list = output.get("assets", [])
#    print(f"Assets found: {assets_list}")

if __name__ == "__main__":
    asyncio.run(main())
