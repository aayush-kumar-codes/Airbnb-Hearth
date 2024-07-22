import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage
from langchain_aws import ChatBedrock
from Agent import AirbnbAgent
from tools.knowledgebase_tool import knowledge_tool
from tools.summarizebase_tool import summarize_tool
from tools.interface_human import human_tool
from tools.summary_keyword_tool import highlight_tool
from tools.compare_tool import compare_tool


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


st.set_page_config(
    page_title="Airbnb Agent",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏠 Airbnb Agent")

if "messages" not in st.session_state:
    st.session_state.messages = [
        ChatMessage(role="assistant", content="How can I help you with your Airbnb-related questions?")]


def clear_session():
    st.session_state.messages = [ChatMessage(role="assistant", content="How can I help you with your Airbnb-related questions?")]


def display_message(role, content):
    if role == "user":
        st.markdown(
            f'<div style="display: flex; justify-content: flex-end;"><div style="background-color: #0084ff; color: white; padding: 10px; border-radius: 20px; max-width: 70%; margin: 5px;">{content}</div></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div style="display: flex; justify-content: flex-start;"><div style="background-color: #f0f0f0; color: black; padding: 10px; border-radius: 20px; max-width: 70%; margin: 5px;">{content}</div></div>',
            unsafe_allow_html=True
        )


chat_container = st.container()
chat_history = []

with chat_container:
    for message in st.session_state.messages:
        display_message(message.role, message.content)

user_input = st.text_input("Type your message here:", key="user_input")
llm = ChatBedrock(
    credentials_profile_name="kamal",
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    model_kwargs={"temperature": 0},
    streaming=True,
    region_name="us-east-1"
)
agent = AirbnbAgent(llm=llm, tools=[knowledge_tool, summarize_tool, human_tool, highlight_tool, compare_tool])

print("!23")
if st.button("Send"):
    if user_input:
        display_message("user", user_input)

        with st.spinner("Thinking..."):

            response_container = st.empty()
            stream_handler = StreamHandler(response_container)
            llm.callbacks = [stream_handler]

            previous = []
            for row in st.session_state.messages:
                print(row)
                previous.append(f"{row.role}: {row.content}")
            response = agent.run(user_input, previous)
            print("Response", response)

        st.session_state.messages.append(ChatMessage(role="user", content=user_input))
        st.session_state.messages.append(ChatMessage(role="assistant", content=response))
        display_message("assistant", response)

        # Rerun the app to clear the input box
        st.rerun()

if st.button("Clear Session"):
    clear_session()
    st.rerun()

st.markdown("---")
st.markdown("👆 Ask me anything about Airbnb listings, bookings, or general information!")
