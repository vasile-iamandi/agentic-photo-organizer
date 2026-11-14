from typing import Dict, List, Any
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage
from graph import build_graph
from langgraph.types import Command

class ImmichAgent:
    """Agent that uses Immich tools through Azure OpenAI and LangGraph HITL."""

    def __init__(self, thread_id: str = "default_user"):
        self.app = build_graph()
        self.thread_id = thread_id

    def _get_config(self):
        return {"configurable": {"thread_id": self.thread_id}}

    # TODO: add run command with graph (this is mw layer: layer between APP and graph)
    async def run(self, message: str) -> Dict[str, Any]:
        """
        Run the graph with a user message.
        Returns dict with:
          - response_text
          - extracted_assets
          - requires_confirmation: bool
          - pending_action: dict description
        """
        config = self._get_config()

        current_state = self.app.get_state(config)
        current_msg_count = len(current_state.values.get("messages", [])) if current_state and current_state.values else 0

        input_data = {"messages": [HumanMessage(content=message)]}
        next_state = await self.app.ainvoke(input_data, config=config)

        next_state = self.app.get_state(config)
        next_messages = current_state.values.get("messages", [])
        new_messages = next_messages[current_msg_count:]


        if next_state.tasks and next_state.tasks[0].interrupts:
            current_interrupt = next_state.tasks[0].interrupts[0]
            interrupt_value = current_interrupt.value
            if interrupt_value.get("action") == "human_confirmation":
                # send confirmation question to application layer
                return self._build_confirmation_question(next_state, new_messages);
        else:
            # regular answer (without tools), just execute
            messages = next_state.get("messages", [])
            assets = self.extract_assets(new_messages)

            final_text = ""
            if messages:
                last_msg = messages[-1]
                final_text = last_msg.content if isinstance(last_msg, (AIMessage, ToolMessage)) else str(last_msg)

            return {
                "response_text": final_text,
                "extracted_assets": assets,
                "pending_action": None
        }

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

    async def resume_confirmation(self, approved: bool) -> Dict[str, Any]:
        """
        Resume graph execution cleanly via conditional edges based on approval decision.
        Also handles chained HITL: if after resuming the graph pauses again at
        human_confirmation (e.g. a second sensitive tool call), returns requires_confirmation=True.
        """
        config = self._get_config()
        current_state = self.app.get_state(config)

        prev_msg_count = len(current_state.values.get("messages", []))
        decision = "approved" if approved else "rejected"

        payload = {
            "approved": approved,
            "note": "User approved operations."
        }
        # Resume execution by sending Command(resume=decision_status) to the interrupted node
        result_state = await self.app.ainvoke(Command(resume=payload), config=config)

        all_messages = result_state.get("messages", [])
        turn_messages = all_messages[prev_msg_count:]
        assets = self.extract_assets(turn_messages)

        last_msg = all_messages[-1] if all_messages else None
        final_text = last_msg.content if isinstance(last_msg, (AIMessage, ToolMessage)) else str(last_msg)

        if approved:
            return {
                "response_text": f"✅ **Action was confirmed and successfully done!**\n\n{final_text}",
                "extracted_assets": assets,
                "requires_confirmation": False
            }
        else:
            return {
                "response_text": f"❌ **Action cancelled by user.**\n\n{final_text}",
                "extracted_assets": [],
                "requires_confirmation": False
            }
