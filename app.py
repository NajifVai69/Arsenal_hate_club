import streamlit as st
from chains import run_banter_chain, run_corner_chain, run_bottle_chain
from agents import run_agent

# --- Page Config ---
st.set_page_config(
    page_title="Arsenal Hate Club",
    page_icon="🔴",
    layout="centered"
)

# --- Header ---
st.title("Arsenal Hate Club 🔴")
st.subheader("AI-powered banter generator — Because someone has to say it.")
st.markdown("> *'We always have corners.' — Mikel Arteta (probably)*")
st.divider()

# --- Sidebar ---
with st.sidebar:
    st.header("Controls")
    mode = st.radio(
        "Choose your weapon:",
        ["Smart Agent (Auto-routes)", "Banter Generator", "Corner Explainer", "Bottle Report"]
    )
    st.divider()

    st.caption("Powered by Groq + LLaMA3 + LangChain")

# --- Main Content ---

if mode == "Smart Agent (Auto-routes)":
    st.markdown("### Smart Banter Agent")
    st.info("The agent will automatically decide whether to roast Arsenal about corners, bottling, or general hopelessness.")
    
    user_input = st.text_area(
        "What's on your mind about Arsenal?",
        placeholder="e.g. 'Tell me about their 2023 season' or 'Why do they love corners so much?'",
        height=100
    )
    
    if st.button("Generate Banter", type="primary"):
        if user_input.strip():
            with st.spinner("Agent thinking... (routing to the right kind of misery)"):
                result = run_agent(user_input)
                st.markdown(f"**Agent routed to:** `{result['route'].upper()}`")
                st.divider()
                st.markdown(result["output"])
        else:
            st.warning("Please enter something first!")

elif mode == "Banter Generator":
    st.markdown("### General Arsenal Banter")
    
    topic = st.selectbox(
        "Pick a topic to roast:",
        ["corner obsession", "title bottling", "Arteta's tactics", "fan delusion",
         "transfer window failures", "VAR conspiracy theories", "the Invincibles era cope"]
    )
    num_banters = st.slider("How many banters?", 1, 10, 5)
    
    if st.button("Roast Them", type="primary"):
        with st.spinner("Loading the disrespect..."):
            output = run_banter_chain(topic=topic, num_banters=num_banters)
            st.markdown(output)

elif mode == "Corner Explainer":
    st.markdown("### Arsenal Corner Situation Analyser")
    st.info("Describe any football situation. We'll explain how Arsenal's answer is... a corner.")
    
    situation = st.text_area(
        "Describe a match situation:",
        placeholder="e.g. 'Arsenal are 1-0 down with 10 minutes left against Bournemouth'",
        height=100
    )
    
    if st.button("Explain the Corner", type="primary"):
        if situation.strip():
            with st.spinner("Calculating corner trajectories..."):
                output = run_corner_chain(situation=situation)
                st.markdown(output)
        else:
            st.warning("Please describe a situation!")

elif mode == "Bottle Report":
    st.markdown("### Arsenal Annual Bottling Report")
    
    season = st.selectbox(
        "Select the season to autopsy:",
        ["2022-23 (The Classic Bottle)", "2023-24 (Sequel Bottle)", 
         "2015-16 (Leicester ruins everything)", "2019-20 (Arteta arrives, chaos ensues)",
         "2007-08 (Eduardo's injury derails everything)"]
    )
    
    if st.button("Generate Bottle Report", type="primary"):
        with st.spinner("Documenting the collapse..."):
            output = run_bottle_chain(season=season)
            st.markdown(output)

# --- Footer ---
st.divider()
st.caption(" For entertainment purposes only. No Gooners were harmed in the making of this app (emotionally, that's on Arteta).")