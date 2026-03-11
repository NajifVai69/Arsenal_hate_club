from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from components.llm import get_llm
from components.prompts import banter_prompt, corner_prompt, bottle_prompt
from components.index import get_retriever

parser = StrOutputParser()
retriever = get_retriever()

def format_docs(docs):
    return "\n".join([d.page_content for d in docs])

def run_banter_chain(topic: str, num_banters: int = 5) -> str:
    llm = get_llm(temperature=0.9)
    chain = banter_prompt | llm | parser
    return chain.invoke({"topic": topic, "num_banters": num_banters})

def run_corner_chain(situation: str) -> str:
    llm = get_llm(temperature=0.9)
    chain = (
        {
            "situation": RunnablePassthrough(),
            "context": retriever | format_docs
        }
        | corner_prompt
        | llm
        | parser
    )
    return chain.invoke(situation)

def run_bottle_chain(season: str) -> str:
    llm = get_llm(temperature=0.9)
    chain = bottle_prompt | llm | parser
    return chain.invoke({"season": season})