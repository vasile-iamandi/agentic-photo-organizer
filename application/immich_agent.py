from typing import Dict, List, Any, AsyncGenerator
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage, AIMessageChunk
from graph import build_graph
from langgraph.types import Command

class ImmichAgent:
    """Agent that uses Immich tools through Azure OpenAI and LangGraph HITL."""

    def __init__(self, thread_id: str = "default_user"):
        self.app = build_graph()
        self.thread_id = thread_id

    def _get_config(self):
        return {"configurable": {"thread_id": self.thread_id}}

    async def _process_stream(self, input_payload: Any, prev_msg_count: int) -> AsyncGenerator[Dict[str, Any], None]:
        config = self._get_config()

        async for mode, chunk in self.app.astream(
            input_payload,
            config=config,
            stream_mode=["messages", "updates"],
            version="v3"
        ):
            if mode == "messages":
                msg_chunk, metadata = chunk
                if isinstance(msg_chunk, AIMessageChunk) and msg_chunk.content and metadata.get("langgraph_node") == "agent":
                    yield {"type": "token", "content": msg_chunk.content}

            elif mode == "updates" and "__interrupt__" in chunk:
                current_state = self.app.get_state(config)
                all_messages = current_state.values.get("messages", [])
                new_messages = all_messages[prev_msg_count:]

                confirmation_data = self._build_confirmation_question(current_state, new_messages)
                yield {
                    "type": "interrupt",
                    "data": confirmation_data
                }
                return

        # 3. Stream finished normally (without interrupt)
        final_state = self.app.get_state(config)
        all_messages = final_state.values.get("messages", [])
        new_messages = all_messages[prev_msg_count:]
        assets = self.extract_assets(new_messages)

        final_text = ""
        if all_messages:
            last_msg = all_messages[-1]
            final_text = last_msg.content if isinstance(last_msg, (AIMessage, ToolMessage)) else str(last_msg)

        yield {
            "type": "final",
            "data": {
                "response_text": final_text,
                "extracted_assets": assets,
                "pending_action": None
            }
        }

    async def run(self, message: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Start conversation."""
        config = self._get_config()
        current_state = self.app.get_state(config)
        prev_msg_count = len(current_state.values.get("messages", [])) if current_state and current_state.values else 0

        input_data = {"messages": [HumanMessage(content=message)]}
        async for event in self._process_stream(input_data, prev_msg_count):
            yield event

    async def resume_confirmation(self, approved: bool) -> AsyncGenerator[Dict[str, Any], None]:
        """Resume execution after user confirmation"""
        config = self._get_config()
        current_state = self.app.get_state(config)
        prev_msg_count = len(current_state.values.get("messages", [])) if current_state and current_state.values else 0

        payload = {
            "approved": approved,
            "note": "User approved operations." if approved else "User rejected operations."
        }

        async for event in self._process_stream(Command(resume=payload), prev_msg_count):
            yield event

    def _build_confirmation_question(self, state, turn_messages: List[BaseMessage]) -> Dict[str, Any]:
        """Build a requires_confirmation response from a state paused at human_confirmation."""
        last_msg = state.values["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", [])

        action_desc = []
        for tc in tool_calls:
            tname = tc.get("name")
            targs = tc.get("args", {})
            if tname == "create_immich_album":
                aname = targs.get("album_name", "Nou")
                aids = targs.get("asset_ids", [])
                action_desc.append(f"Creare album: **{aname}** cu {len(aids)} poze.")
            elif tname == "add_assets_to_immich_album":
                action_desc.append(f"Adăugare {len(targs.get('asset_ids', []))} poze în albumul `{targs.get('album_id')}`.")
            else:
                action_desc.append(f"Executare `{tname}`.")

        assets = self.extract_assets(turn_messages)

        return {
            "response_text": "⚠️ **Este necesară confirmarea ta pentru a continua această acțiune:**\n" + "\n".join(f"- {d}" for d in action_desc),
            "extracted_assets": assets,
            "pending_action": {
                "tool_calls": tool_calls,
                "description": action_desc
            }
        }

    def extract_assets(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """Extract assets returned by tools."""
        extracted_assets = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                content = str(msg.content)
                if "Found " in content and "assets:" in content:
                    try:
                        start_idx = content.find("[{")
                        end_idx = content.rfind("}]")
                        if start_idx != -1 and end_idx != -1:
                            import ast
                            parsed = ast.literal_eval(content[start_idx:end_idx + 2])
                            if isinstance(parsed, list):
                                extracted_assets.extend(parsed)
                    except Exception:
                        pass
        return extracted_assets
