import asyncio

from dotenv import load_dotenv
load_dotenv()
import immich_tools
from graph import build_graph
from langgraph.types import Command
from immich_agent import ImmichAgent

async def main():
    immich_agent = ImmichAgent()

    user_message = "Creeaza un album si pune o poza cu pisici!"

    ai_output = await immich_agent.run(user_message)
    pending_action = ai_output.get("pending_action") or {}
    pending_tools = pending_action.get("tool_calls")

    if (pending_tools):
        print(f"There are tools which shall be confirmed: {pending_tools}")
        await immich_agent.resume_confirmation(approved=True)
    print(f"response_text: {ai_output.get("response_text")}")

def display_confirmation_message(payload):
    None

if __name__ == "__main__":
    asyncio.run(main())
