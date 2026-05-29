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

import json, re

from dotenv import load_dotenv
from tool_lib import processing_execution_tools

import time

begin = time.time()


load_dotenv()


class ProcessingState(TypedDict):
    selected_tools: List[str]
    supervisor_instructions: List[Any]
    processing_to_supervisor_msgs: List[Any]
    messages: List[Any]


def prepare_resources_for_retrieval():
    """Prepare resources for retrieval

    Returns:
        resources: a dictionary containing all the tools
    """

    from pathlib import Path
    import importlib.util
    import sys

    tool_descriptions = []

    base_dir = Path(__file__).resolve().parent
    desc_dir = base_dir / "tool_lib" / "processing_code_description"

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

    # Use retrieval to get relevant resources
    resources = {
        "tools": all_tools,
    }

    return resources


def prompt_based_tool_retrieval(supervisor_instructions, resources, current_tools):

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
            ..., description="Whether the current tool set must be updated now based on the Supervisor’s reply."
        )
        rationale: str = Field(
            ..., description="Brief reason (1 sentence) justifying the decision, referencing the Supervisor’s reply."
        )


    def _should_update_toolset(supervisor_instructions, resources, current_tools) -> bool:
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

            #resources["tools"] is a list of dictionaries, each dictionary contains the information of a tool
            for tool in resources["tools"]:
                if tool["name"] in current_tools:
                    selected_tools_info.append(tool)

            tool_str_list = []
            for tool in selected_tools_info:
                tool_str_list.append(f"name: {tool['name']}\ndescription: {tool['description']}")

            selected_tools_str = "\n".join(tool_str_list)

            latest_instruction = _latest_text(supervisor_instructions)

            parser = JsonOutputParser(pydantic_object=_UpdateDecision)
            format_instructions = parser.get_format_instructions()

            whether_toolset_update_prompt = f"""
You are a neuroimaging preprocessing expert collaborating with a Supervisor.
You follow the Supervisor’s instructions and execute preprocessing using neuroimaging processing tools.

## Task
You have just reported your latest progress to the Supervisor, and the Supervisor has responded. Based on the Supervisor’s reply, decide whether you need to UPDATE your current tool set now in order to proceed with the next preprocessing steps.

### Tools you can currently use:

{selected_tools_str}

### The Supervisor’s reply you are handling NOW:

{latest_instruction}

## Decision guideline
- ONLY update your current tool set if the Supervisor’s reply explicitly mentions using new tools or replacing existing tools. In all other cases, you do NOT need to update the current tool set.

## Output format
**Your output must follow the JSON schema below.**

{format_instructions}
            """

            resp = decider_llm.invoke(whether_toolset_update_prompt)
            try:
                data = parser.parse(resp.content or "")

                return bool(data.get("update"))
            except Exception:
                sys.exit("Error in _should_update_toolset")


    def select_or_update_tools(supervisor_instructions, resources, current_tools) -> List[str]:
        class SelectedResources(BaseModel):
            selected_tools: List[str] = Field(
                ..., description="The names of the selected tools. Must be exactly the same as the names in the available tools list."
            )

        parser = JsonOutputParser(pydantic_object=SelectedResources)
        format_instructions = parser.get_format_instructions()

        # 将tools转换为字符串格式，方便插入到prompt中。
        all_tools = resources["tools"]
        all_tools_str = "\n".join([f"- [{tool['name']}]: {tool['description']}" for tool in all_tools])
        latest_instruction = _latest_text(supervisor_instructions)


        prompt = f"""
You are a neuroimaging preprocessing expert collaborating with a Supervisor.
You follow the Supervisor’s instructions and execute preprocessing using neuroimaging processing tools.

## Task
You have just reported your latest progress to the Supervisor, and the Supervisor has responded. Based on the Supervisor’s reply, select from the full tool catalog the tools you need and UPDATE your current tool set so you can proceed with the next preprocessing steps.

### Tools you currently have in hand:

{current_tools}

Note: If the list above is empty, interpret it as “no tools currently in hand”.

### Full tool catalog (name: short description):

{all_tools_str}

### The Supervisor’s reply you are handling NOW:

{latest_instruction}

## Guideline
- You may only update or add tools that are explicitly indicated by the Supervisor’s reply. Do NOT remove tools unrelated to the Supervisor’s instruction. For example, if the Supervisor asks to replace the skull-stripping tool, restrict your update to skull-stripping related tools you already have; do NOT change tools used for other steps.

## Output format
**Your output must follow the JSON schema below.**

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
        except Exception:
            sys.exit("Error in prompt_based_retrieval")

        selected_resources = {
            "selected_tools": selected_tools,
        }

        return selected_resources


    need_update = _should_update_toolset(supervisor_instructions, resources, current_tools)
    if need_update:
        selected_resources = select_or_update_tools(supervisor_instructions, resources, current_tools)
    else:
        selected_resources = {"selected_tools": current_tools}


    print("\n" + "-"*50)
    print(f"Selected Resources: {selected_resources}")
    print("-"*50 + "\n")

    return selected_resources




def construct_system_prompt_with_selected_resources(all_resources, selected_resources):
    """
    Construct the system prompt with the selected resources for the processing agent.
    """
    # Convert the selected resources to strings

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

    # construct the system prompt with the selected tools
    processing_system_prompt = f"""
You are a neuroimaging preprocessing expert. Your mission is to execute concrete preprocessing steps for sMRI/fMRI under the direction of the Supervisor Agent, using the tools that are currently available to you.

# Your basic behavioral pattern is to complete the corresponding neuroimaging preprocessing tasks by writing and running the appropriate Python scripts and submitting jobs via `sbatch`.

Specifically, the Python script should import and call the corresponding processing tool functions according to the requirements, and be written as a complete Python script. The processing logic in the Python script must be designed for a single subject, or in other words, be subject-agnostic. That is, this script should be a general script applicable to processing all subjects.
In particular:

* When writing the Python script, if you need to import existing Python code (functions or model classes), you must use absolute imports. The specific import path should be determined according to the `module` field described in the codebase. For example, if the module field is `tool_lib.fsl`, then the import should be written as `from tool_lib.fsl import tool_name`.
* After each processing step finishes execution, the corresponding tool function will return the execution log of that tool. You need to concatenate the execution logs of all tools to form a complete execution log, **and at the end of the script you must use a `print()` statement to print the complete log.**

You are performing image processing on a shared Linux server. In order to obtain the required resources, when submitting the Python script for image processing, you must use SLURM and submit it via `sbatch`. Specifically, to run the corresponding neuroimaging processing script, you first need to write a `.sh` script. You can refer to the following template to write this script:


```bash
#!/bin/bash
#SBATCH --job-name={{job_name}}
#SBATCH --array=1-{{number_of_subjects}}%20 # for example, 100 subject will be 1-100%20
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:0
#SBATCH --partition=h200day
#SBATCH --output=Path/temp_workspace/slurm_outputs/{{job_name}}-%A_%a.out 

SUBJECT_LIST={{Path_to_subject_IDs}}/subjects.txt
SUB_ID=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "$SUBJECT_LIST")

LOG_DIR=Path/temp_workspace/logs
mkdir -p "$LOG_DIR"

# Write each subject's processing log into the corresponding log file, rather than the original slurm output
exec >"${{LOG_DIR}}/${{SUB_ID}}.log" 2>&1

# Command to run the python script
python path_to_python_script.py --subject_id ${{SUB_ID}}
```

The`subjects.txt`in the script is a **list of subject IDs**, one subject ID per line, for example:
```text
sub-subid_1
sub-subid_2
sub-subid_3
```
You need to create the corresponding `subjects.txt` file in advance according to the actual data situation.


After writing the '.sh' bash script, you need to submit this script via `sbatch --wait` using the `run_bash_command` tool. For example, `sbatch --wait path_to_bash_script.sh`.

---

# Important Guidelines:

* Your workspace directory is `Path/temp_workspace`, and **all processing outputs and intermediate results must be placed in this directory**.
* Before starting to process, you should first think carefully to ensure that the tools currently available to you are sufficient to complete the task according to the supervisor’s requirements (if multiple tools are needed, ensure that the tools are compatible with each other). If they are not sufficient, you must stop and inform the supervisor so that they can adjust the preprocessing pipeline.
* Basically, you should follow the workflow below:


1. First, write the Python script and use this script to process **a set of sampled subjects (for example, 10 subjects)** to test the validity of the script.
2. Check the results of these subjects to see whether any expected derivatives files are missing.
 
 - If the any expected derivatives files are missing, check the script or logs. Fix any issues if found. If the script is correct, report the issue concisely to the Supervisor Agent for guidance.
 - If none of the expected derivatives files are missing, you must stop and report the completed processing steps and generated data to the Supervisor Agent for quality control (QC). Tell the supervisor that you just finished processing few sampled subjects and ask for confirmation of the quality of the generated data before processing all subjects using the same pipeline.
 - Wait until the supervisor confirms the quality of the generated data. If data quality issues are reported, revise the preprocessing pipeline according to the Supervisor’s instructions.
 - If the Supervisor Agent confirms the data quality is acceptable or confirms a final pipeline, the preprocessing pipeline is finalized. Then run the script in batch for all subjects.

3. After all subjects have been processed, write a simple script to check whether any expected derivatives files are missing for all subjects. Note that you must not check subjects one by one manually; instead, you should use a script to perform this check.
4. Stop and Report the final preprocessing pipeline, as well as the storage locations of the generated data, to the Supervisor Agent. Inform the Supervisor Agent that you have finished your job and any downstream analysis can proceed.


* Note that the log file for each subject may be empty, as some tools do not generate logs during execution. Therefore, the core criterion for determining whether processing was performed correctly is whether the expected derivatives files exist.
* You **must** use the `--wait` parameter when submitting sbatch tasks using `sbatch --wait`.
* Before submitting an `sbatch` script, the directory specified in `#SBATCH --output` **must be created in advance**; otherwise, the job **will fail to run**.
* When fixing a failed neuroimaging preprocessing step, do not rerun the entire pipeline by default—first detect which steps have already completed successfully and only rerun the failed step and its true downstream dependencies, unless a full rerun is genuinely necessary.
* When requesting supervisor to perform quality control, you do not need to tell supervisor what specific QC checks to perform, supervisor will make the decision by itself.
* If the workflow or data flow specified by the Supervisor (e.g., the input data for a step) conflicts with the tool instructions provided to you (for example, the tool requires skull-in data but the Supervisor asks you to use skull-stripped data), you must stop and report the conflict to the Supervisor so they can make the necessary adjustments. You must never process images in a way that violates the tool instructions.
* When the supervisor asks you to switch to a new pipeline for processing the subject, if some intermediate outputs in the new pipeline are exactly the same as those in a pipeline you have already processed before, then for the new pipeline you do not need to call the tools again to repeat the same processing. You can directly copy the previously generated intermediate outputs into the new pipeline directory directly.


---

# Available Codebases (Codebases that can be imported to complete the task):

{selected_tools_str}
    """
    

    print("@"*30)
    print(processing_system_prompt)
    print("@"*30 + "\n\n")

    return processing_system_prompt




def processing_agent_node(state: ProcessingState):
    processing_agent_llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), 
    model="gpt-5.2"      
    )   

    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + ">>> ENTERING PROCESSING AGENT <<<".center(78) + "#")
    print("#" + " "*78 + "#")
    print("#"*80 + "\n")
    

    all_resources = prepare_resources_for_retrieval()
    selected_resources = prompt_based_tool_retrieval(state["supervisor_instructions"], all_resources, state["selected_tools"])
    processing_system_prompt = construct_system_prompt_with_selected_resources(all_resources, selected_resources)

    processing_agent = create_react_agent(
        model=processing_agent_llm,
        tools=processing_execution_tools,
        prompt=processing_system_prompt)

    # Compose runtime messages from persistent processing context and latest supervisor instructions
    runtime_messages = (state.get("messages", []) or []) + ([state.get("supervisor_instructions", [])[-1]] if state.get("supervisor_instructions") else [])

    # Stream the agent; after loop, take the final full transcript as updated messages
    prev_len = len(runtime_messages)
    final_messages = runtime_messages

    start_time = time.time()
    for step in processing_agent.stream({"messages": runtime_messages}, stream_mode="values"):
        step_messages = step["messages"]
        final_messages = step_messages

        print('\n' +'*'*5 + '[Processing Agent]' + '*'*5)
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

    # Append only the final report message to processing_to_supervisor_msgs
    existing_to_supervisor = state.get("processing_to_supervisor_msgs", []) or []
    
    if not last_ai_msg:
        sys.exit("Error in processing_agent.py: last_ai_msg is None")

    updated_to_supervisor = existing_to_supervisor + [last_ai_msg]

    return {
        "messages": final_messages,
        "processing_to_supervisor_msgs": updated_to_supervisor,
        "selected_tools": selected_resources["selected_tools"],
    }




ProcessingGraph = StateGraph(ProcessingState)

ProcessingGraph.add_node("processing_agent_node", processing_agent_node)
ProcessingGraph.add_edge(START, "processing_agent_node")
ProcessingGraph.add_edge("processing_agent_node", END)


ProcessingAgent = ProcessingGraph.compile()



