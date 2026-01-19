
# ----------------------------------------------------------------------
# Dependencies
# ----------------------------------------------------------------------

from langgraph.graph import StateGraph, START, END 
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage, message_to_dict, messages_to_dict
from langgraph.graph.message import add_messages
from langchain_core.prompts import PromptTemplate
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command
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

    chapter_number: Annotated[
        Optional[int],
        Field(..., title="chapter number")
    ] = None

    rag_query: Annotated[ 
        Optional[str],
        Field(..., title="RAG query", description="It will used to fetch documents for generating quiz")
    ] = None

    quiz: Annotated[ 
        Optional[ str | dict ],
        Field(..., title="Generated Quiz")
    ] = None








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




def orchestrator(state: SkoolAgentState) -> Literal['quiz','end']:
    human_msg = get_latest_human_message(state.messages).content

    sys_msg =  """ 
    You are an expert intent classifier.
    Determine whether the user is asking for, requesting, or referring to a quiz.
    If yes, respond only with quiz.
    If not, respond only with end.
    Do not explain or add anything else.
    """
    response = llm_client.chat.completions.create(
        model = "gpt-4o-mini",
        messages=[
            {
                'role': 'system',
                'content': sys_msg
            },
            {
                'role': 'user',
                'content': human_msg
            }
        ], 
        max_tokens=16000
    )

    return response.choices[0].message.content





def extract_chapter_number(state: SkoolAgentState):

    human_msg = get_latest_human_message(state.messages).content
    chapters = load_chapters(Configurations.chapters_file_path)

    template_msg = """
    you are an expert data extractor
    Your task is to extract the chapter number from the user query and provided chapters list,

    chapters list:
    {chapters}

    Instructions:
    - user may provide either chapter name or number 
    - user provided chapter information must be available in the chapters list
    - if chapter name/number successfully found in the provided chapter list 
    - if user provide valid chapter name instead number then you have to find valid chapter number (from chapters_list)
    - return only a number (interger) not string
    - if no valid chapter name/number found then simple return None (exactly)

    output must be a valid number (integer) - chapter index/number
    """
    template = PromptTemplate(template=template_msg, input_variables=['chapters'])
    sys_msg = template.invoke({'chapters':chapters}).text 

    response = llm_client.chat.completions.create(
        model = "gpt-4o-mini",
        messages=[
            {
                'role': 'system',
                'content': sys_msg
            },
            {
                'role': 'user',
                'content': human_msg
            }
        ], 
        max_tokens=4096
    )

    response = response.choices[0].message.content
    try:
        if response == 'None':
            ch_num = None
        else:
            ch_num = int(response)
    except ValueError:
        return { 'chapter_number': None }
    
    return  { 'chapter_number': ch_num }






def quiz_info_validation(state: SkoolAgentState):

    ch_num = state.chapter_number 

    chapters_list = list(load_chapters(Configurations.chapters_file_path).values())

    # interruption
    data_msg = {
        "chapter": ch_num,
        "message": "Please specify chapter first.",
        "chapters_index": chapters_list,
        "instruction": "Select a chapter from the provided list"
        # "selected_chapter_number": None
    }

    decision = interrupt(data_msg)

    try:
        if isinstance(ch_num, int) or isinstance(decision["selected_chapter_number"], int):

            chap_num = isinstance(ch_num, int) or isinstance(decision["selected_chapter_number"], int)

            QUERY_TEMPLATE = """
            Retrieve content from the knowledge base that is specifically related to the chapter:

            Chapter Name: "{chapter_name}"

            Your task is to fetch the most relevant passages, sections, explanations, definitions, examples, and connected concepts that clearly help understand this chapter.

            Match the following:
            - Exact or partial overlap with the chapter name
            - Synonyms or rephrasings of the chapter title
            - Any subtopics, exercises, summaries, or descriptions related to the chapter
            - Relevant definitions, diagrams, explanations, or conceptual context
            - Content where this topic is indirectly discussed

            Avoid:
            - Unrelated chapters
            - General random content

            Return only the content most relevant to the chapter concept.
            """

            rag_query = QUERY_TEMPLATE.format(chapter_name=chapters_list[chap_num])

            return { "rag_query": rag_query }
        
    except KeyError:
        pass    # It means chapter number is not provided of fetched 
                # and decision["selected_chapter_number"] doesn't exists

    
    
    if decision["selected_chapter_number"] == None:
        return {"messages": [AIMessage(content="Quiz not generate due to unclearity of chapter.")]}

        









template_content = """
You are an AI quiz generation agent.

Your task is to generate EXACTLY 5-10 multiple-choice questions (MCQs)** 
using ONLY the provided chapter content.

====================
STRICT INSTRUCTIONS
====================
1. Use only facts, definitions, or statements explicitly present in the chapter content.
2. Do NOT use outside knowledge or assumptions.
3. If information is missing or unclear, do not invent it.
4. Each MCQ must contain:
   - One clear question
   - Exactly 4 options
   - Exactly 1 correct option
5. The correct option must be directly supported by the chapter content.
6. Keep the language simple, factual, and unambiguous.
7. Question level will be basic.
8. Try to be short options.

====================
OUTPUT EXPECTATION
====================
Return the result in a structure compatible with the following models:

- List[MCQ]
- Each MCQ contains:
  - question: string
  - options: list of strings (length = 4)
  - right_option: string (must exactly match one option)

Do NOT include explanations, markdown, comments, or extra text.


====================
OUTPUT STRUCTURE
====================
[
    {{
        "question": "---",
        "options": [ "op1", "op2", "op3", "op4"],
        "right_option": [ --- ]
    }},
    {{
        "question": "---",
        "options": [ "op1", "op2", "op3", "op4"],
        "right_option": [ --- ]
    }},
    ...
]


====================
CHAPTER CONTENT
====================
{chapter_context}


Now you can start you work.
"""

def query_generation_node(state: SkoolAgentState):

    # fetching docs 
    docs = fm.mmr_search(
        name=Configurations.vector_store_name, 
        query=state.rag_query, 
        k=8
    )

    context = ""
    for doc in docs:
        context += doc.page_content 

    template = PromptTemplate(
        template=template_content,
        input_variables=['chapter_context'] 
    )

    sys_msg = template.invoke({'chapter_context':context}).text

    response = llm_client.chat.completions.create(
        model = "gpt-4o-mini",
        messages=[
            {
                'role': 'user',
                'content': sys_msg
            }
        ], 
        max_tokens=8192
    )

    try:
        quiz_content = parse_python_like_string(response.choices[0].message.content)
        return { "quiz": quiz_content }
    except Exception as e:
        pass 
    
    return { "quiz": quiz_content }

    







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
    workflow.add_node("rag", rag_node)
    workflow.add_node("agent_chat", chat_node)
    workflow.add_node("ch_name_extractor", extract_chapter_number)
    workflow.add_node("quiz_info_validator", quiz_info_validation)
    workflow.add_node("quiz_generator", query_generation_node)

    # connecting nodes 
    workflow.set_entry_point("rag")
    workflow.add_edge("rag","agent_chat")
    workflow.add_conditional_edges(
        source="agent_chat",
        path=orchestrator,
        path_map={
            'end': END,
            'quiz': "ch_name_extractor"
        }
    )
    workflow.add_edge("ch_name_extractor", "quiz_info_validator")
    workflow.add_edge("quiz_info_validator", "quiz_generator")
    workflow.set_finish_point("quiz_generator")

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
        total_score = 4 * len(quiz_mcqs)

        for index,ques in enumerate(quiz_mcqs):
            if answers[index] == ques['right_option']:
                score += 4
            else:
                score -= 1

        st.write(f"You got {score}/{total_score}, +4 mark for each right answer and -1 for each wrong answer.")







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
                
                if final_state['quiz']:
                    if isinstance(final_state['quiz'], dict):
                        quiz_dialog(final_state['quiz'])
                    else:
                        logging.error("Quiz generated but not popuped may be improper format either in LLM generation or quiz extraction")

                # condition for interruption
                if "__interrupt__" in final_state:
                    # it means interruption occur
                    chapters_names_list = final_state['__interrupt__'][0].value['chapters_index']
                    chapters_names = []
                    for index, name in enumerate(chapters_names_list):
                        chapters_names.append(f"{index}) {name}")

                    selected_chapter = st.selectbox(
                        label = "Chapters list",
                        options=chapters_names,
                        label_visibility="hidden",
                        placeholder="Select a chapter",
                    )

                    selected_chapter_number = selected_chapter.split(")")

                    resumed_final_state =  st.session_state.chatbot_agent.invoke(
                        Command(resume={"selected_chapter_number": selected_chapter_number}),
                        config=config,
                    )

                    if resumed_final_state['quiz']:
                        if isinstance(resumed_final_state['quiz'], dict):
                            quiz_dialog(resumed_final_state['quiz'])
                        else:
                            logging.error("[interrupted:resumed] Quiz generated but not popuped may be improper format either in LLM generation or quiz extraction")



                else: # normal chatbot response
                    # fetching ai message
                    ai_msg = final_state['messages'][-1].content

                    # Save bot message
                    st.session_state.messages.append({"role": "assistant", "content": ai_msg})

            # Display bot message
            with st.chat_message("assistant"):

                st.write_stream(stream_text(ai_msg), cursor="🤖")









if __name__ == "__main__":
    working_page()
















