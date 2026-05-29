from tool_lib import data_awareness_tools

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, List, Any
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
import os
import sys

from dotenv import load_dotenv

import time




load_dotenv()

class DataAwarenessState(TypedDict):
    selected_tools: List[str]
    supervisor_instructions: List[Any]
    data_awareness_to_supervisor_msgs: List[Any]
    messages: List[Any]



def data_awareness_node(state: DataAwarenessState):
    data_aware_llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), 
    model="gpt-5.2"      
    )   

    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + ">>> ENTERING DATA AWARENESS AGENT <<<".center(78) + "#")
    print("#" + " "*78 + "#")
    print("#"*80 + "\n")
    

    data_awareness_system_prompt = """
You are a neuroimaging data profiling expert. Your task is to profile the data based on another supervisor agent's needs using the appropriate tools and return the relevant information to the supervisor agent.

# IMPORTANT RULES:
- You are a large language model-based neuroimaging data profiling expert agent, when you are collecting information or returning results to the supervisor agent, you should pay attention to the potential problem of information token explosion for LLM.
- While ensuring accuracy and completeness of information, you may appropriately summarize and consolidate the content throughout the process.

    """

    new_selected_tools = data_awareness_tools

    # print(f'len(new_selected_tools): {len(new_selected_tools)}')
    # print("Selected tool set: ", new_selected_tools)

    data_awareness_agent = create_react_agent(
        model=data_aware_llm,
        tools=new_selected_tools,
        prompt=data_awareness_system_prompt)

    # Compose runtime messages from persistent processing context and latest supervisor instructions
    runtime_messages = (state.get("messages", []) or []) + ([state.get("supervisor_instructions", [])[-1]] if state.get("supervisor_instructions") else [])

    # Stream the agent; after loop, take the final full transcript as updated messages
    prev_len = len(runtime_messages)
    final_messages = runtime_messages

    for step in data_awareness_agent.stream({"messages": runtime_messages}, stream_mode="values"):
        start_time = time.time()
        step_messages = step["messages"]
        final_messages = step_messages

        print('\n' +'*'*5 + '[Data Awareness Agent]' + '*'*5)
        step_messages[-1].pretty_print()
        print(f'this step took {time.time() - start_time} seconds')

        # used for debugging
        # print('@'*30)
        # print(step)
        # print('@'*30 + '\n\n')


    # Determine the last assistant-authored message produced in this run (from the delta)
    delta = final_messages[prev_len:] if len(final_messages) > prev_len else []
    last_ai_msg = None
    for m in reversed(delta):
        if getattr(m, "type", "") == "ai":
            last_ai_msg = m
            break

    # Append only the final report message to data_awareness_to_supervisor_msgs
    existing_to_supervisor = state.get("data_awareness_to_supervisor_msgs", []) or []

    if not last_ai_msg:
        sys.exit("Error in data_awareness_agent.py: last_ai_msg is None")

    updated_to_supervisor = existing_to_supervisor + [last_ai_msg]

    new_selected_tools = [t.name for t in new_selected_tools]
    
    return {
        "messages": final_messages,
        "data_awareness_to_supervisor_msgs": updated_to_supervisor,
        "selected_tools": new_selected_tools,
    }



DataAwarenessGraph = StateGraph(DataAwarenessState)

DataAwarenessGraph.add_node("data_awareness_node", data_awareness_node)
DataAwarenessGraph.add_edge(START, "data_awareness_node")
DataAwarenessGraph.add_edge("data_awareness_node", END)


DataAwarenessAgent = DataAwarenessGraph.compile()











