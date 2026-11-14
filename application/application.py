import asyncio
from dotenv import load_dotenv
load_dotenv()

from immich_agent import ImmichAgent

async def handle_stream(stream_generator):
    """Help function which consumes any stream from agent"""
    pending_confirmation = None

    async for event in stream_generator:
        event_type = event["type"]

        if event_type == "token":
            # Display each instant text chunk, without new line
            print(event["content"], end="", flush=True)

        elif event_type == "interrupt":
            pending_confirmation = event["data"]
            print(f"\n\n{pending_confirmation['response_text']}")

        elif event_type == "final":
            print("\n--- Execution finished ---")
            if event["data"]["extracted_assets"]:
                print(f"Poze găsite: {len(event['data']['extracted_assets'])}")

    return pending_confirmation

async def main():
    agent = ImmichAgent()
    user_message = "Creeaza un album si pune o poza cu pisici!"

    print("User: ", user_message, flush=True)
    print("Agent: ", end="", flush=True)

    # 1. Start stream
    confirmation_needed = await handle_stream(agent.run(user_message))

    # 2. If graph stopped at human_confiramtion, ask approval and resume the stream
    while confirmation_needed:
        pending_tools = confirmation_needed.get("pending_action", {}).get("tool_calls")
        print(f"Unelte în așteptare: {pending_tools}")

        # Aproove: (Yes / No)
        user_approved = True
        print("\n✅ Action approved...\n🤖 Agent: ", end="", flush=True)

        # Resume execution and consume new stream (use while for the case when another interrupt occurs )
        confirmation_needed = await handle_stream(agent.resume_confirmation(approved=user_approved))

if __name__ == "__main__":
    asyncio.run(main())
