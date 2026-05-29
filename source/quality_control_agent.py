from tool_lib import qc_tools

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
import time
from dotenv import load_dotenv



load_dotenv()

class QualityControlState(TypedDict):
    selected_tools: List[str]
    selected_knowledge_docs: List[str]
    supervisor_instructions: List[Any]
    quality_control_to_supervisor_msgs: List[Any]
    messages: List[Any]




def prepare_resources_for_retrieval():
    """Prepare resources for retrieval

    Returns:
        resources: a dictionary containing all the tools and know-how documents
    """

    from pathlib import Path
    import importlib.util
    import sys

    tool_descriptions = []

    base_dir = Path(__file__).resolve().parent
    desc_dir = base_dir / "tool_lib" / "quality_control_description"

    if desc_dir.is_dir():
        for py_file in desc_dir.glob("*.py"):
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            if spec is None or spec.loader is None:
                sys.exit(f"Failed to load module {py_file.stem}")

            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)  # type: ignore[arg-type]
            except Exception:
                sys.exit(f"Failed to load module {py_file.stem}")

            tool_descriptions.extend(module.descriptions)

    all_tools = tool_descriptions

    knowledge_documents = []

    kb_dir = base_dir / "knowledge_base" / "quality_control"

    if kb_dir.is_dir():
        for md_file in kb_dir.glob("*.md"):
            content = md_file.read_text(encoding="utf-8")

            lines = content.splitlines()

            knowledge_name = md_file.stem
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("# "):
                    knowledge_name = stripped[2:].strip()
                    break

            overview_lines = []
            in_overview = False
            for line in lines:
                if line.startswith("## ") and "Overview" in line and not in_overview:
                    in_overview = True
                    continue
                if in_overview and line.startswith("## "):
                    break
                if in_overview:
                    overview_lines.append(line)

            knowledge_documents.append(
                {
                    "knowledge_name": knowledge_name,
                    "knowledge_overview": "\n".join(overview_lines).strip(),
                    "knowledge_content": content,
                }
            )

    # Use retrieval to get relevant resources
    resources = {
        "tools": all_tools,
        "knowledge_docs": knowledge_documents,
    }

    return resources




def prompt_based_retrieval(supervisor_instructions, resources, current_tools, current_knowledge_docs):

    from pydantic import BaseModel, Field
    from typing import List
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_openai import ChatOpenAI
    import os
    import sys
    
    def _latest_text(msgs: List) -> str:
        for m in reversed(msgs or []):
            return getattr(m, "content", "") or ""


    class _UpdateDecision(BaseModel):
        update: bool = Field(
            ..., description="Whether the current tool set or knowledge documents must be updated now based on the Supervisor’s reply."
        )
        rationale: str = Field(
            ..., description="Brief reason (1 sentence) justifying the decision, referencing the Supervisor’s reply."
        )


    def _should_update_toolset(supervisor_instructions, resources, current_tools, current_knowledge_docs) -> bool:
        decider_llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-5.2"
            )
        

        need_update = False

        if len(current_tools) == 0:
            need_update = True
            return need_update
        else:
            # construct the string for selected tools
            selected_tools_info = []
            for tool in resources["tools"]:
                if tool["name"] in current_tools:
                    selected_tools_info.append(tool)

            tool_str_list = []
            for tool in selected_tools_info:
                tool_str_list.append(f"name: {tool['name']}\ndescription: {tool['description']}")

            selected_tools_str = "\n".join(tool_str_list)


            # construct the string for selected knowledge documents
            selected_knowledge_docs_info = []
            for doc in resources["knowledge_docs"]:
                if doc["knowledge_name"] in current_knowledge_docs:
                    selected_knowledge_docs_info.append(doc)

            selected_knowledge_docs_str = "\n".join([f"- [{doc['knowledge_name']}]: {doc['knowledge_overview']}" for doc in selected_knowledge_docs_info])


            latest_instruction = _latest_text(supervisor_instructions)

            parser = JsonOutputParser(pydantic_object=_UpdateDecision)
            format_instructions = parser.get_format_instructions()

            whether_toolset_or_knowledge_docs_update_prompt = f"""
You are a neuroimaging quality control expert collaborating with a Supervisor.
You follow the Supervisor’s instructions and execute specific quality control tasks using neuroimaging quality control tools.

The neuroimaging quality control genearlly consists of two stages: before preprocessing QC (using MRIQC) and after preprocessing QC (using specific QC tools) for several key preprocessing steps.
In each stage, the QC consists of two parts: metric-based automated QC and visual-inspection-based further verfication QC.


## Task
You have just reported your latest progress to the Supervisor, and the Supervisor has responded. Based on the Supervisor’s reply, decide whether you need to UPDATE your current tool set or knowledge documents (if any) now in order to proceed with the next quality control steps.

### Tools you currently have in hand:

{selected_tools_str}

### Knowledge documents you currently have in hand (it could be empty):

{selected_knowledge_docs_str}

### The Supervisor’s latest instruction you are handling NOW:

{latest_instruction}

## Decision guideline
- ONLY update your current tool set or knowledge documents if the Supervisor’s reply explicitly mentions using new tools, replacing existing tools or requiring new tasks that may require new knowledge documents. In all other cases, you do NOT need to update the current tool set or knowledge documents.

## Output format
**Your output must follow the JSON schema below.**

{format_instructions}
            """

            resp = decider_llm.invoke(whether_toolset_or_knowledge_docs_update_prompt)
            try:
                data = parser.parse(resp.content or "")

                return bool(data.get("update"))
            except Exception:
                sys.exit("Error in _should_update_toolset")


    def inner_prompt_based_retrieval(supervisor_instructions, resources, current_tools, current_knowledge_docs):

        from pydantic import BaseModel, Field
        from typing import List
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_openai import ChatOpenAI
        import os
        import sys


        class SelectedResources(BaseModel):
            selected_tools: List[str] = Field(
                ..., description="The names of the selected tools. Must be exactly the same as the names in the available tools list."
            )
            selected_knowledge_docs: List[str] = Field(
                ..., description="The names of the selected knowledge documents. Must be exactly the same as the names in the available knowledge documents list."
            )

        parser = JsonOutputParser(pydantic_object=SelectedResources)
        format_instructions = parser.get_format_instructions()

        # 将tools和knowledge_docs转换为字符串格式，方便插入到prompt中。
        all_tools = resources["tools"]
        all_knowledge_docs = resources["knowledge_docs"]
        all_tools_str = "\n".join([f"- [{tool['name']}]: {tool['description']}" for tool in all_tools])
        all_knowledge_docs_str = "\n".join([f"- [{doc['knowledge_name']}]: {doc['knowledge_overview']}" for doc in all_knowledge_docs])

        # construct the string for selected tools
        selected_tools_info = []
        for tool in resources["tools"]:
            if tool["name"] in current_tools:
                selected_tools_info.append(tool)

        tool_str_list = []
        for tool in selected_tools_info:
            tool_str_list.append(f"name: {tool['name']}\ndescription: {tool['description']}")

        selected_tools_str = "\n".join(tool_str_list)

        # construct the string for selected knowledge documents
        selected_knowledge_docs_info = []
        for doc in resources["knowledge_docs"]:
            if doc["knowledge_name"] in current_knowledge_docs:
                selected_knowledge_docs_info.append(doc)

        selected_knowledge_docs_str = "\n".join([f"- [{doc['knowledge_name']}]: {doc['knowledge_overview']}" for doc in selected_knowledge_docs_info])


        latest_instruction = _latest_text(supervisor_instructions)

        prompt = f"""
You are an expert neuroimaging quality control assistant. Your task is to select the relevant resources to help complete a given quality control task.
You follow instructions from a supervisor and select the appropriate tools or knowledge documents according to the supervisor’s instructions. Currently, the tools and knowledge documents you already have in hand are as follows:

## Tools you currently have in hand:
{selected_tools_str}

## Knowledge documents you currently have in hand (it could be empty):
{selected_knowledge_docs_str}

Note: If the list above is empty, interpret it as “no tools or knowledge documents currently in hand”.

---
Now, the supervisor has given you new instructions. Based on that new instruction, select from all available tools and knowledge documents as needed, and update your current set of tools or knowledge documents so that you can proceed with the required quality control steps.


##The Supervisor’s latest instruction you are handling NOW:

{latest_instruction}
---

# All Available Resources:

Below are the available resources. For each category, select items that are directly or indirectly relevant to the task.

The neuroimaging quality control genearlly consists of two stages: before preprocessing QC (using MRIQC) and after preprocessing QC (using specific QC tools) for several key preprocessing steps.
In each stage, the QC consists of two parts: metric-based automated QC and visual-inspection-based further verfication QC.

Be generous in your selection - include resources that might be useful for the task, even if they're not explicitly mentioned in the task description.

Each resource is represented in the format of "[name]: description".

It is possible that the task requires none of the available resources. In this case, respond with an empty list.

## ALL AVAILABLE TOOLS (Code that can be used to complete the task):
{all_tools_str}

## ALL AVAILABLE KNOWLEDGE DOCUMENTS (Guidebooks & Protocols), you may not need any provided knowledge documents if the task not relevant to any knowledge documents:
{all_knowledge_docs_str}

# Output format:
Your output must follow the JSON schema below:
{format_instructions}
        """

        selector_llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-5.2"
            )
        
        resp = selector_llm.invoke(prompt)

        try:
            data = parser.parse(resp.content or "")
            selected_tools = data.get("selected_tools")
            selected_knowledge_docs = data.get("selected_knowledge_docs")
        except Exception:
            sys.exit("Error in prompt_based_retrieval")

        selected_resources = {
            "selected_tools": selected_tools,
            "selected_knowledge_docs": selected_knowledge_docs,
        }

        print("\n" + "-"*50)
        print(f"Selected resources: {selected_resources}")
        print("-"*50 + "\n")


        return selected_resources


    need_update = _should_update_toolset(supervisor_instructions, resources, current_tools, current_knowledge_docs)
    if need_update:
        selected_resources = inner_prompt_based_retrieval(supervisor_instructions, resources, current_tools, current_knowledge_docs)
    else:
        selected_resources = {"selected_tools": current_tools, "selected_knowledge_docs": current_knowledge_docs}


    print("\n" + "-"*50)
    print(f"Selected Resources: {selected_resources}")
    print("-"*50 + "\n")

    return selected_resources





def construct_system_prompt_with_selected_resources(all_resources, selected_resources):
    """
    Construct the system prompt with the selected resources for the quality control agent.
    """
    # Convert the selected resources to strings

    # 每个selected tools的string格式类似如下：形式
    """
    name: tool_name
    description: tool_description
    module: module_name
    detailed_schema: detailed_schema
    """
    # construct the string for selected tools
    selected_tools_info = []

    #resources["tools"] is a list of dictionaries, each dictionary contains the information of a tool
    for tool in all_resources["tools"]:
        if tool["name"] in selected_resources["selected_tools"]:
            selected_tools_info.append(tool)

    tool_str_list = []
    for tool in selected_tools_info:
        tool_str_list.append(f"name: {tool['name']}\ndescription: {tool['description']}\nmodule: {tool['module']}\ndetailed_schema: {tool['detailed_schema']}")

    selected_tools_str = "\n".join(tool_str_list)

    # construct the string for selected knowledge documents
    selected_knowledge_docs_info = []
    for doc in all_resources["knowledge_docs"]:
        if doc["knowledge_name"] in selected_resources["selected_knowledge_docs"]:
            selected_knowledge_docs_info.append(doc)

    knowledge_doc_str_list = []
    for doc in selected_knowledge_docs_info:
        knowledge_doc_str_list.append(doc['knowledge_content'])

    selected_knowledge_docs_str = "\n".join(knowledge_doc_str_list)

    # write system prompt
    system_prompt = f"""
You are a neuroimaging quality control expert. Your task is to perform quality control (QC) on neuroimaging data as requested.

You will collaborate with a supervisor and a neuroimaging processing agent to complete QC tasks. The supervisor communicates directly with you. They will inform you of the current QC task and where the data are stored. If you need additional data that the supervisor has not provided, you should first ask the supervisor. You may also check the file system if necessary, but asking the supervisor has priority. The neuroimaging processing agent is responsible for actually processing the data.

In general, quality control consists of two stages:

---

### 1. Before Preprocessing Quality Control

This stage takes place before any actual data processing.

At this stage, you will primarily use MRIQC-related tools to perform QC on raw, unprocessed data. The basic workflow is:

1. Run MRIQC to obtain Image Quality Metrics (IQMs) and corresponding visual inspection outputs for each subject.
2. Use the metrics to identify subjects with abnormal values.
3. Perform visual inspection only on the small subset of subjects flagged as abnormal.
4. For each subject, review only the most critical images and provide a final judgment.

At the end of this stage, you must report the before-preprocessing QC results to the supervisor:
Which subjects have data quality that is too poor and should be excluded from further processing, while the rest of the subjects can proceed to subsequent processing and analysis.

---

### 2. After Preprocessing Quality Control

This stage takes place after the neuroimaging processing agent has completed data processing.

Your task is to evaluate whether the processed data are of sufficient quality for downstream analysis. This stage also includes both metric-based and visual-inspection-based QC.

* If the number of subjects is small (e.g., when the processing agent is testing the preprocessing pipeline on a small subset < 10 subjects before large-scale processing), you only need to perform visual inspection.
* If the number of subjects is large (e.g., all subjects in the dataset have been processed), you should:

  1. For each preprocessing step that requires QC, compute the metrics relevant to that specific step only.
  2. For each preprocessing step separately, identify outlier subjects based only on that step's own metrics (e.g., the most abnormal 15% for that step).
  3. For each preprocessing step separately, perform visual inspection only on the subjects flagged for that same step.
  4. A subject may therefore receive visual QC for one step but not for another, depending on which step-specific metric screen flagged that subject.
  5. After the step-specific visual inspections are completed, aggregate the per-step QC decisions into the final subject-level judgment and clearly report which preprocessing step(s) failed for each rejected subject.
---

Note that the neuroimaging processing pipeline may involve many different steps. You only need to perform QC for the specific processing steps that the supervisor tells you were performed by the processing agent. Use the tools and codebase available to you to conduct the appropriate QC. Do not perform unnecessary additional checks.

You must report QC results to the supervisor. Tell the supervisor which subjects have data quality that is poor and provide concise reason. The supervisor will relay your feedback to the processing agent, who will make adjustments accordingly.

---

To achieve this, you will be using an python coding environment equipped with a variety of tools and resources (codebases) to assist you throughout the process.

# General and Important Instructions:
* You must use the tool functions I provide for any visual or metric-based quality control (QC). You may not write your own code to calculate metrics.
* All inputs to the QC tool functions must strictly follow their parameter requirements. During QC, the Supervisor should specify where the required data is located, but they may forget to mention some locations. In that case, do not guess and do not generate any synthetic data yourself. You must pause and ask the Supervisor for the exact data locations before continuing the QC. If the required data is ultimately not provided, skip the corresponding QC step.
* Given a task, make a plan first using the `write_todos` tool. The plan should be specific, concise and detailed.
* During the process of completing the task, all script files you write and all intermediate results must be stored in the designated Workspace directory: `Path/temp_workspace`.
* When writing the code, always print out the step status in a clear and concise manner, like a research log. For example, at the end of each step, print('step xx has completed'). Otherwise the system will not be able to know what has been done.
* If you need to import existing python code (function or model class), you must use absolute imports. The specific import path should be determined according to the `module` field described in the codebase. For example, if module field is `tool_lib.quality_control_lib`, then the import should be written as `from tool_lib.quality_control_lib import tool_name`.
* You should prioritize using the provided python codebases instead of writing your own code.
* You are a large language model-based neuroimaging quality control expert agent, when gathering information, always be mindful of potential token explosion, especially when there are many subjects (more than 20) to inspect. Ensure completeness and accuracy while keeping the amount of information under control.
* When writing Python code for **after-preprocessing quality control**, you must pay attention to efficiency. When handling large datasets, prioritize writing parallelized code.

* Specifically:

  * When computing after-processing QC metrics (which are primarily CPU-intensive tasks), you should use `concurrent.futures.ProcessPoolExecutor`.
  * When performing visual inspection (which is not CPU-intensive), you should use `concurrent.futures.ThreadPoolExecutor`.
  * In all cases, `max_workers` should set to 5.

* All visual inspection Python tools return results in the same format: a Python dictionary with the following structure:

```python
{{
    "verdict": Literal["ACCEPTABLE", "REJECTED"],  # ACCEPTABLE if the QC visualization is acceptable; otherwise REJECTED
    "reject_reason": Optional[str]  # Required only when verdict == "REJECTED". A concise reason based on observation.
}}
```

---

# Below are the resources you may refer to and use to complete the task:

## Available Codebases (Codebases that can be imported to complete the task):

{selected_tools_str}
    """

    return system_prompt



def quality_control_node(state: QualityControlState):
    quality_control_llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), 
    model="gpt-5.2"      
    )   


    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + ">>> ENTERING QUALITY CONTROL AGENT <<<".center(78) + "#")
    print("#" + " "*78 + "#")
    print("#"*80 + "\n")

    all_resources = prepare_resources_for_retrieval()
    selected_resources = prompt_based_retrieval(state["supervisor_instructions"], all_resources, state["selected_tools"], state["selected_knowledge_docs"])
    quality_control_system_prompt = construct_system_prompt_with_selected_resources(all_resources, selected_resources)


    print("^"*50)
    print(f"Quality Control System Prompt: {quality_control_system_prompt}")
    print("^"*50 + "\n")
    
    quality_control_agent = create_react_agent(
        model=quality_control_llm,
        tools=qc_tools,
        prompt=quality_control_system_prompt)


    # Compose runtime messages from persistent processing context and latest supervisor instructions
    runtime_messages = (state.get("messages", []) or []) + ([state.get("supervisor_instructions", [])[-1]] if state.get("supervisor_instructions") else [])

    # Stream the agent; after loop, take the final full transcript as updated messages
    prev_len = len(runtime_messages)
    final_messages = runtime_messages

    start_time = time.time()
    for step in quality_control_agent.stream({"messages": runtime_messages}, stream_mode="values"):
        step_messages = step["messages"]
        final_messages = step_messages

        print('\n' +'*'*5 + '[Quality Control Agent]' + '*'*5)
        step_messages[-1].pretty_print()
        print(f'this step took {time.time() - start_time} seconds')
        start_time = time.time()

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

    # Append only the final report message to downstream_analysis_to_supervisor_msgs
    existing_to_supervisor = state.get("quality_control_to_supervisor_msgs", []) or []

    if not last_ai_msg:
        sys.exit("Error in quality_control_agent.py: last_ai_msg is None")

    updated_to_supervisor = existing_to_supervisor + [last_ai_msg]
    
    return {
        "messages": final_messages,
        "quality_control_to_supervisor_msgs": updated_to_supervisor,
        "selected_tools": selected_resources["selected_tools"],
        "selected_knowledge_docs": selected_resources["selected_knowledge_docs"],
    }




QualityControlGraph = StateGraph(QualityControlState)

QualityControlGraph.add_node("quality_control_node", quality_control_node)
QualityControlGraph.add_edge(START, "quality_control_node")
QualityControlGraph.add_edge("quality_control_node", END)


QualityControlAgent = QualityControlGraph.compile()



