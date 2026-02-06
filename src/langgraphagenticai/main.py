import streamlit as st

from src.langgraphagenticai.ui.streamlitui.loadui import LoadStreamlitUI
from src.langgraphagenticai.LLMS.groqllm import GroqLLM
from src.langgraphagenticai.graph.graph_builder import GraphBuilder
from src.langgraphagenticai.ui.streamlitui.display_result import DisplayResultStreamlit


def load_langgraph_agenticai_app():
    # -------------------- UI LOAD --------------------
    ui = LoadStreamlitUI()
    user_input = ui.load_streamlit_ui()

    if not user_input:
        st.error("Failed to load UI inputs.")
        return

    usecase = user_input.get("selected_usecase")

    # -------------------- USER INPUT --------------------
    user_message = None

    if usecase == "AI News":
        # AI News uses button + dropdown
        if st.session_state.get("IsFetchButtonClicked"):
            user_message = st.session_state.get("timeframe")
    else:
        # Other usecases use chat input
        user_message = st.chat_input("Enter your message")

    # If no input yet, stop execution (Streamlit rerun safe)
    if not user_message:
        return

    # -------------------- LLM SETUP --------------------
    try:
        llm_config = GroqLLM(user_contols_input=user_input)
        model = llm_config.get_llm_model()

        if not model:
            st.error("LLM model initialization failed.")
            return
    except Exception as e:
        st.error(f"LLM setup error: {e}")
        return

    # -------------------- GRAPH SETUP --------------------
    try:
        graph_builder = GraphBuilder(model)
        graph = graph_builder.setup_graph(usecase)
    except Exception as e:
        st.error(f"Graph setup failed: {e}")
        return

    # -------------------- DISPLAY RESULT --------------------
    DisplayResultStreamlit(
        usecase=usecase,
        graph=graph,
        user_message=user_message
    ).display_result_on_ui()
