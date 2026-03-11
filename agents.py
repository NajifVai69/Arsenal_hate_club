from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from typing import TypedDict, Literal
from components.llm import get_llm
from chains import run_banter_chain, run_corner_chain, run_bottle_chain

llm = get_llm(temperature=0.3)

# --- Agent State ---
class AgentState(TypedDict):
    user_input: str
    route: str
    output: str


# --- Router Node ---
def router_node(state: AgentState) -> AgentState:
    """Decides which chain to invoke based on user input."""
    prompt = f"""You are a routing agent for the Arsenal Hate Club app.
Based on the user's input, choose ONE of these routes:
- "banter" → general Arsenal mockery and jokes
- "corner" → jokes about Arsenal's corner obsession
- "bottle" → Arsenal's infamous title collapses

User input: "{state['user_input']}"
Respond with ONLY one word: banter, corner, or bottle."""

    response = llm.invoke([HumanMessage(content=prompt)])
    route = response.content.strip().lower()
    if route not in ["banter", "corner", "bottle"]:
        route = "banter"
    return {**state, "route": route}


# --- Execution Nodes ---
def banter_node(state: AgentState) -> AgentState:
    output = run_banter_chain(topic=state["user_input"], num_banters=4)
    return {**state, "output": output}


def corner_node(state: AgentState) -> AgentState:
    output = run_corner_chain(situation=state["user_input"])
    return {**state, "output": output}


def bottle_node(state: AgentState) -> AgentState:
    output = run_bottle_chain(season=state["user_input"])
    return {**state, "output": output}


# --- Route Decider ---
def decide_route(state: AgentState) -> Literal["banter_node", "corner_node", "bottle_node"]:
    return f"{state['route']}_node"


# --- Build LangGraph ---
def build_agent():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("banter_node", banter_node)
    graph.add_node("corner_node", corner_node)
    graph.add_node("bottle_node", bottle_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", decide_route)
    graph.add_edge("banter_node", END)
    graph.add_edge("corner_node", END)
    graph.add_edge("bottle_node", END)

    return graph.compile()


arsenal_agent = build_agent()


def run_agent(user_input: str) -> dict:
    result = arsenal_agent.invoke({
        "user_input": user_input,
        "route": "",
        "output": ""
    })
    return result