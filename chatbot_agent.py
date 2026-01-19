
# ----------------------------------------------------------------------
# Dependencies
# ----------------------------------------------------------------------

from langgraph.graph import StateGraph, START, END 
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage, message_to_dict, messages_to_dict
from langgraph.graph.message import add_messages
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field  
from openai import AzureOpenAI
from dotenv import load_dotenv
import streamlit as st
import sqlite3
import logging

from utils import *
from rag import SchoolRAG
from config import Configurations
from faiss_manager import FaissManager

from typing import Literal, Optional, List, Dict, Tuple, Annotated
import os 
import uuid
import shutil







##############################################################################
#
#       Before deploying it:
#           1) set load_pre_stored = False (while creating rag instance)
#
##############################################################################




fm = FaissManager(root_dir="LOCAL")




logging.basicConfig(
    level=logging.INFO, # Set the minimum level to log
    format='%(asctime)s - %(levelname)s - %(message)s' # Define message format
)





# ----------------------------------------------------------------------
# secret keys and database setup (for persistency)
# ----------------------------------------------------------------------

load_dotenv()

# creating database (sqlite)
database = sqlite3.connect(Configurations.checkpointer_sqlite_path, check_same_thread=False)

# checkpointer
checkpointer = SqliteSaver(conn=database)






# ----------------------------------------------------------------------
# Sidebar and page configurations
# ----------------------------------------------------------------------

st.header("Skool Chatbot", divider=True)

st.sidebar.header("About Skool Chatbot")







# ----------------------------------------------------------------------
# LLM client and graph state
# ----------------------------------------------------------------------

llm_client = AzureOpenAI()


class SkoolAgentState(BaseModel):

    messages: Annotated[List[BaseMessage], add_messages]

    raw_content: Annotated[
        Optional[str], 
        Field(..., title="Raw content", description="this content fetched from the rag (vector store)")
    ] = ""








# ----------------------------------------------------------------------
# Agent nodes
# ----------------------------------------------------------------------


def rag_node(state: SkoolAgentState):

    human_msg = get_latest_human_message(state.messages)

    # fetching docs from rag
    docs = fm.mmr_search(name=Configurations.rag_vector_store_path, query=human_msg.content, k=2)
    context = ""
    for doc in docs:
        context += doc.page_content

    return { 'raw_content': context }

    


def chat_node(state: SkoolAgentState):
    history_messages = state.messages
    history_messages.append(
        SystemMessage(f"you have to answer only from provided `raw_content` don't try to answer \
                      apart from `raw_content`, If you don't know then simple say i don't know. \
                      \n`raw_content`: {state.raw_content}"
        )
    )
    # logging.info(msg=f"raw_content: {state.raw_content}")
    messages = to_openai_messages(history_messages)

    response = llm_client.chat.completions.create(
        model = "gpt-4o-mini",
        messages=messages, 
        max_tokens=16000
    )

    message = AIMessage(content=response.choices[0].message.content)

    return { "messages": [message] }








def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)







# ----------------------------------------------------------------------
# Building workflow
# ----------------------------------------------------------------------
def build_chatbot():

    # workflow
    workflow = StateGraph(SkoolAgentState)

    # defining nodes
    workflow.add_node("agent_chat", chat_node)
    workflow.add_node("rag", rag_node)

    # connecting nodes 
    workflow.set_entry_point("rag")
    workflow.add_edge("rag","agent_chat")
    workflow.set_finish_point("agent_chat")

    logging.info("Building agent's instance...")

    # building workflow agent
    agent = workflow.compile(checkpointer=checkpointer)

    return agent





def stream_text(text: str) :
    for word in text.split(" "):
        yield f" {word}"




@st.dialog("Quiz")
def quiz_dialog(quiz_mcqs):
    answers = list()
    for index,mcq in enumerate(quiz_mcqs):
        # st.write(f"{index+1}) {mcq['question']}")
        ans = st.radio(
            f"{index+1}) {mcq['question']}",
            mcq['options'],
            index=None
        )

        answers.append(ans)

    if st.button("Submit"):
        score = 0
        total_score = 10

        for index,ques in enumerate(quiz_mcqs):
            if answers[index] == ques['right_option']:
                score += 2

        st.write(f"You got {score}/{total_score}")







# **************************************** utility functions *************************

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['messages'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    state = st.session_state.chatbot_agent.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])




def working_page() -> None:
    st.set_page_config(page_title="Skool Chatbot | Working", page_icon='🤖')

    # # ----------------------------------------------------------- Sidebar --------------------------------------------------------------- #
    # st.sidebar.header('Menu', divider=True)
    # st.sidebar.write('It is basically a RAG (Retrieval Augmented Generation) based web application, \
    #                 that is optimized for chatting with any PDF in an efficient way.')
    


    # st.sidebar.subheader("Connect with me!")
    # st.sidebar.write("[Kaggle](https://www.kaggle.com/architty108)")
    # st.sidebar.write("[Github](https://www.github.com/a4archit)")
    # st.sidebar.write("[LinkedIn](https://www.linkedin.com/in/a4archit)")

    # if st.sidebar.button('Go to Home', type='primary'):
    #     st.session_state.page = 'home'




    # **************************************** Session Setup ******************************
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    if 'thread_id' not in st.session_state:
        st.session_state['thread_id'] = generate_thread_id()

    if 'chat_threads' not in st.session_state:
        st.session_state['chat_threads'] = retrieve_all_threads()

    add_thread(st.session_state['thread_id'])


    # ******************************* Sidebar threads system *********************


    if st.sidebar.button('New Chat'):
        reset_chat()

    with st.spinner(text="Loading Chats", show_time=True):
        st.sidebar.header('My Conversations')

        for thread_id in st.session_state['chat_threads'][::-1]:
            if st.sidebar.button(f"{str(thread_id)[:15]}..."):
                try:
                    if st.session_state.chatbot_agent is None:
                        st.write("Upload a PDF first before access chats.")
                        return None 
                except AttributeError:
                    st.write("Upload a PDF first before access chats.")
                    return None
                
                st.session_state['thread_id'] = thread_id
                messages = load_conversation(thread_id)

                temp_messages = []

                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        role='user'
                    elif isinstance(msg, AIMessage):
                        role='assistant'
                    else:
                        role='system'
                    temp_messages.append({'role': role, 'content': msg.content})

                st.session_state['messages'] = temp_messages


    # --------------------------------------------------------- Body ------------------------------------------------------------------- #

    # Initialize chat mode if not set
    if "chat_mode" not in st.session_state:
        st.session_state.chat_mode = False

    # If not chatting yet, show uploader + button
    if not st.session_state.chat_mode:
        # st.header('Skool Chatbot', divider=True)

        uploaded_file = st.file_uploader(
            label = "Upload PDF",
            type = 'pdf',
            accept_multiple_files = False,
            help = "You can upload your PDF file here."
        )

        if uploaded_file:

            if st.button(label = 'Chat with this PDF', type='primary'):
                with st.spinner(text="Saving file..."):
                    # Switch to chat mode
                    st.session_state.chat_mode = True
                    # Store the uploaded file for later use if needed
                    st.session_state.uploaded_file = uploaded_file
                    # Clear old messages if any
                    st.session_state.messages = []

                    # rag instance
                    st.session_state.rag = None 

                    # vector store
                    st.session_state.vector_store = None 

                    # You can read it directly
                    bytes_data = uploaded_file.read()

                    # Or save it to a file to get a path
                    with open("user_uploaded_file.pdf", "wb") as f:
                        f.write(bytes_data)


                with st.spinner("Setting up RAG...", show_time=True):
                    # generate chunks of pdf
                    st.session_state.rag = SchoolRAG(
                        pdf_path=Configurations.rag_vector_store_path, 
                        vector_store_name=Configurations.rag_vector_store_path,
                        load_pre_stored=True
                    )
                    st.session_state.rag.setup(load_pre_stored=True)

                    # logging.info("Loading vector store... ")
                    # st.session_state.vector_store = fm.load_store(
                    #     name=Configurations.rag_vector_store_path
                    # )
                    


                with st.spinner(text="Setting up chatbot..."):
                    st.session_state.chatbot_agent = build_chatbot()


               


    

    # If in chat mode, show chat
    if st.session_state.chat_mode:
        st.header('Skool Chatbot Chat')

        # Try another PDF button
        if st.button('📄 Try another PDF'):
            # Reset state to go back to upload mode
            st.session_state.chat_mode = False
            st.session_state.uploaded_file = None
            st.session_state.messages = []

            # rag.delete_all_vector_stores()
            store_path = Configurations.rag_vector_store_path
            if os.path.exists(store_path):
                shutil.rmtree(store_path)

            # Stop here to prevent rendering the rest
            st.rerun()

        

        # Initialize session state for messages
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # loading the conversation history
        for message in st.session_state['messages']:
            with st.chat_message(message['role']):
                st.text(message['content'])

        # Chat input box
        prompt = st.chat_input("Type your message...")

        # When user submits a message
        if prompt:
            # Save user message
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user"):
                st.write(prompt)

            with st.spinner("Thinking...", show_time=True):
                
                # -------------------- calling agent here ------------------------
                initial_state = SkoolAgentState(messages=[HumanMessage(prompt)])
                # getting response
                config = {'configurable': {'thread_id': st.session_state['thread_id']}}
                final_state = st.session_state.chatbot_agent.invoke(initial_state, config=config)
                
                ai_msg = final_state['messages'][-1].content

                try:
                    quiz_mcqs = final_state['quiz']['mcqs']
                    quiz_dialog(quiz_mcqs)

                except Exception as e:
                    logging.info("No MCQs fetch from Agent's Final State.")
            
                # Save bot message
                st.session_state.messages.append({"role": "assistant", "content": ai_msg})

            # Display bot message
            with st.chat_message("assistant"):

                st.write_stream(stream_text(ai_msg), cursor="🤖")









if __name__ == "__main__":
    working_page()
















