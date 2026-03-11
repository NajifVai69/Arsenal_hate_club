from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.documents import Document

# Arsenal banter knowledge base
ARSENAL_FACTS = [
    "Arsenal have already scored 16 Premier League goals from corners this season, matching the all time league record with 9 games still to play.",
    "Arsenal blew a 5-point lead with 5 games to go in the 2022-23 Premier League season.",
    "Mikel Arteta once said corners are 'like having an extra player' — they conceded 3 on the day.",
    "Arsenal fans declare 'this is our year' every single August since 2005.",
    "Arsenal went the entire second half of the season without winning away from home in 2024.",
    "Arsenal takes an average of 44 seconds to set up a corner, which is the longest in the Premier League, leading some to joke that they are spending the time perfecting their evil genius plans.",
    "Since the start of the 2023/24 season, Arsenal have scored a staggering 36 Premier League goals from corners, which is 15 more than any other team.",
    "They finished 2nd in 2023 after being first for most of the season — classic Arsenal.",
    "Arsenal fans genuinely believe VAR is specifically targeting them.",
]

def get_retriever():
    """
    Vector Index: FAISS in-memory store of Arsenal banter facts.
    Uses FakeEmbeddings for demo (swap with real embeddings in production).
    """
    docs = [Document(page_content=fact) for fact in ARSENAL_FACTS]
    embeddings = FakeEmbeddings(size=1536)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})