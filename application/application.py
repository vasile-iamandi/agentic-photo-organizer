import asyncio

from dotenv import load_dotenv
load_dotenv()
import immich_tools
from graph import build_graph
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.types import Command

async def main():
    graph = build_graph()
    config = {"configurable": {"thread_id": "thread_name"}}
    input = [HumanMessage(content="Creeaza un album si pune o poza cu pisici!")]
    output = await graph.ainvoke({"messages" : input}, config)

    if "__interrupt__" in output:
        from_llm = output["__interrupt__"][0]
        print(f"Graph interrupted: {from_llm}")

        resume_command={"approved": True, "note": "Command has been approved."}
        print(f"Resume interrupt with command {resume_command}")
        from_llm = await graph.ainvoke(Command(resume=resume_command), config)

        # Print AI and tools messages
        for message in from_llm.get("messages", []):
            if isinstance(message, AIMessage) and message.content:
                print(f"\n🤖 AI: {message.content}")
            elif isinstance(message, ToolMessage):
                print(f"\n🛠️ Răspuns de la tool ({message.name}):")
                print(message.content)

    else:
        print(f"Normal Execution (without tools): output: {output}")

if __name__ == "__main__":
    asyncio.run(main())
