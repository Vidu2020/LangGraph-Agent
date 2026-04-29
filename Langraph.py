from dotenv import load_dotenv
load_dotenv()

import os
from typing import Literal
from typing_extensions import NotRequired

from serpapi import GoogleSearch

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage, HumanMessage
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage

from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver


SupportStep = Literal[
    "warranty_collector",
    "issue_classifier",
    "resolution_specialist",
]


class SupportState(AgentState):
    current_step: NotRequired[SupportStep]
    warranty_status: NotRequired[Literal["in_warranty", "out_of_warranty"]]
    issue_type: NotRequired[Literal["hardware", "software"]]


@tool
def web_search(query: str) -> str:
    """Search the web with SerpApi and return the top 3 results."""
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return "SERPAPI_API_KEY is missing in your .env file."

    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    output = []
    for item in results.get("organic_results", [])[:3]:
        title = item.get("title", "")
        link = item.get("link", "")
        snippet = item.get("snippet", "")
        output.append(f"Title: {title}\nLink: {link}\nSnippet: {snippet}")

    return "\n\n".join(output) if output else "No results found."


@tool
def record_warranty_status(
    status: Literal["in_warranty", "out_of_warranty"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """Record the warranty status and move to issue classification."""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Warranty status recorded as: {status}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "warranty_status": status,
            "current_step": "issue_classifier",
        }
    )


@tool
def record_issue_type(
    issue_type: Literal["hardware", "software"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """Record the issue type and move to resolution."""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Issue type recorded as: {issue_type}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "issue_type": issue_type,
            "current_step": "resolution_specialist",
        }
    )


@tool
def provide_solution(solution: str) -> str:
    """Provide a troubleshooting or support resolution to the user."""
    return f"Solution provided: {solution}"


@tool
def escalate_to_human(reason: str) -> str:
    """Escalate the case to a human support agent."""
    return f"Escalating to human support. Reason: {reason}"


STEP_CONFIG = {
    "warranty_collector": {
        "prompt": """You are a device support agent.
Ask whether the device is under warranty.
When the user answers, call record_warranty_status.""",
        "tools": [record_warranty_status],
        "requires": [],
    },
    "issue_classifier": {
        "prompt": """You are a device support agent.
Warranty status: {warranty_status}

Ask the user to describe the issue briefly.
If the user mentions physical damage, cracks, broken parts, or display damage, call record_issue_type with "hardware".
If the user mentions app crashes, slowness, bugs, software glitches, or settings problems, call record_issue_type with "software".
If the description is unclear, ask one short clarifying question.""",
        "tools": [record_issue_type],
        "requires": ["warranty_status"],
    },
    "resolution_specialist": {
        "prompt": """You are a device support agent.
Warranty status: {warranty_status}
Issue type: {issue_type}

If the issue is software, provide troubleshooting using provide_solution.
If the issue is hardware and in warranty, explain the repair or replacement path using provide_solution.
If the issue is hardware and out of warranty, call escalate_to_human.
If the user asks for an authorized service center, use web_search to find it.""",
        "tools": [provide_solution, escalate_to_human, web_search],
        "requires": ["warranty_status", "issue_type"],
    },
}


@wrap_model_call
def apply_step_config(request: ModelRequest, handler) -> ModelResponse:
    current_step = request.state.get("current_step", "warranty_collector")
    step_config = STEP_CONFIG[current_step]

    for key in step_config["requires"]:
        if request.state.get(key) is None:
            raise ValueError(f"{key} must be set before reaching {current_step}")

    request = request.override(
        system_prompt=step_config["prompt"].format(**request.state),
        tools=step_config["tools"],
    )
    return handler(request)


model = init_chat_model("gpt-4o-mini")

agent = create_agent(
    model=model,
    tools=[
        record_warranty_status,
        record_issue_type,
        provide_solution,
        escalate_to_human,
        web_search,
    ],
    state_schema=SupportState,
    middleware=[apply_step_config],
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "support-thread-1"}}

def print_last_ai_message(result):
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            print("\nAssistant:", msg.content)
            return
    print("\nAssistant: (no final AI response found)")

if __name__ == "__main__":
    print("Device Support Agent (LangGraph + SerpApi)")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        result = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )
        print_last_ai_message(result)

        # Optional: debug current state each turn
        state = agent.get_state(config)
        print("\n[DEBUG state]", {
            "current_step": state.values.get("current_step"),
            "warranty_status": state.values.get("warranty_status"),
            "issue_type": state.values.get("issue_type"),
        })