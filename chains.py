from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from components.llm import get_llm
from components.prompts import banter_prompt, corner_prompt, bottle_prompt
from components.index import get_retriever

llm = get_llm(temperature=0.9)
parser = StrOutputParser()

# --- Chain 1: Banter Generator Chain ---
banter_chain = banter_prompt | llm | parser

# --- Chain 2: Corner Explainer Chain (with retrieval) ---
retriever = get_retriever()

def format_docs(docs):
    return "\n".join([d.page_content for d in docs])

corner_chain = (
    {
        "situation": RunnablePassthrough(),
        "context": retriever | format_docs
    }
    | corner_prompt
    | llm
    | parser
)

# --- Chain 3: Bottle Report Chain ---
bottle_chain = bottle_prompt | llm | parser


def run_banter_chain(topic: str, num_banters: int = 5) -> str:
    return banter_chain.invoke({"topic": topic, "num_banters": num_banters})


def run_corner_chain(situation: str) -> str:
    return corner_chain.invoke(situation)


def run_bottle_chain(season: str) -> str:
    return bottle_chain.invoke({"season": season})