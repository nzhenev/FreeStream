import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.chains import ConversationalRetrievalChain
from utility_funcs import configure_retriever, StreamHandler, PrintRetrievalHandler, set_llm

# Initialize LangSmith tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "FreeStream-v2"
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets.LANGCHAIN.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = st.secrets.LANGCHAIN.LANGCHAIN_API_KEY

# Set up page config
st.set_page_config(page_title="FreeStream: Free AI Tooling", page_icon="🗣️📄")
st.title("FreeStream")
st.header(":rainbow[_Empowering Everyone with Advanced AI Tools_]", divider="red")
st.caption(":violet[_Democratizing access to advanced AI tools like GPT-3.5-turbo, offering a free service to simplify document retrieval and generation._]")
st.sidebar.subheader("__User Panel__")

# Add a way to upload files
uploaded_files = st.sidebar.file_uploader(
    label="Upload a PDF or text file",
    type=["pdf", "doc", "docx", "txt"],
    help="Types supported: pdf, doc, docx, txt",
    accept_multiple_files=True
)
if not uploaded_files:
    st.info("Please upload documents to continue.")
    st.stop()

retriever = configure_retriever(uploaded_files)

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

# Create a dictionary with keys to chat model classes
model_names = {
    "ChatOpenAI GPT-3.5 Turbo": ChatOpenAI(
        model_name="gpt-3.5-turbo-0125",
        openai_api_key=st.secrets.OPENAI.openai_api_key,
        temperature=0.7,
        streaming=True
    ),
}

selected_model = st.selectbox(
    label="Choose your chat model:",
    options=list(model_names.keys()),
    key="model_selector",
    on_change=set_llm
)

# Load the selected model dynamically
llm = model_names[selected_model]

# Create a chain that ties everything together
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory,
    verbose=True
)

# if the length of messages is 0, or when the user \
    # clicks the clear button,
    # show a default message from the AI
if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    # show a default message from the AI
    msgs.add_ai_message("How can I help you?")

# Display coversation history window
avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

# Display user input field and enter button
if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)
            
    # Display assistant response
    with st.chat_message("assistant"):
        # Check for the presence of the "messages" key in session state
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])
        st.toast('Success!', icon="✅")