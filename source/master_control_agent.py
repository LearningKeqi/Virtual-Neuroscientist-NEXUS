from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.types import Command
from typing import List, Any, Annotated
from pydantic import Field

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool, InjectedToolCallId
from langchain_community.callbacks import get_openai_callback

from langchain_openai import ChatOpenAI

from processing_agent import ProcessingAgent
from quality_control_agent import QualityControlAgent
from data_awareness_agent import DataAwarenessAgent
from downstream_analysis_agent import DownstreamAnalysisAgent

from dotenv import load_dotenv
import os

load_dotenv()

# Global variables to track token usage across all sub-agents
global_sub_agent_tokens = {
    "processing": {"total": 0, "prompt": 0, "completion": 0, "cost": 0.0},
    "quality_control": {"total": 0, "prompt": 0, "completion": 0, "cost": 0.0},
    "data_awareness": {"total": 0, "prompt": 0, "completion": 0, "cost": 0.0},
    "downstream_analysis": {"total": 0, "prompt": 0, "completion": 0, "cost": 0.0}
}


def _count_messages_tokens_safe(chat_model: ChatOpenAI, messages: List[Any]) -> int:
    """Best-effort token counter for a list of BaseMessages.

    Uses `chat_model.get_num_tokens_from_messages` (which also accounts for
    per-message role overhead). Falls back to a `chars // 4` estimate if the
    underlying tokenizer cannot handle the current model (e.g. a model name
    not yet recognized by tiktoken).
    """
    messages = messages or []
    try:
        return chat_model.get_num_tokens_from_messages(messages)
    except Exception:
        total_chars = 0
        for m in messages:
            content = getattr(m, "content", "") or ""
            if isinstance(content, list):
                for c in content:
                    total_chars += len(str(c))
            else:
                total_chars += len(str(content))
        return max(0, total_chars // 4)


class SupervisorState(AgentState):
    user_request: str = ""

    # Processing Agent State
    processing_agent_msgs: List[Any]
    processing_agent_selected_tools: List[str]
    supervisor_to_processing_instructions: List[Any]
    processing_to_supervisor_msgs: List[Any]

    # Quality Control Agent State
    quality_control_agent_msgs: List[Any]
    quality_control_agent_selected_tools: List[str]
    quality_control_agent_selected_knowledge_docs: List[str]
    supervisor_to_quality_control_instructions: List[Any]
    quality_control_to_supervisor_msgs: List[Any]

    # Data Awareness Agent State
    data_awareness_agent_msgs: List[Any]
    data_awareness_agent_selected_tools: List[str]
    supervisor_to_data_awareness_instructions: List[Any]
    data_awareness_to_supervisor_msgs: List[Any]

    # Downstream Analysis Agent State
    downstream_analysis_agent_msgs: List[Any]
    downstream_analysis_agent_selected_tools: List[str]
    downstream_analysis_agent_selected_knowledge_docs: List[str]
    supervisor_to_downstream_analysis_instructions: List[Any]
    downstream_analysis_to_supervisor_msgs: List[Any]




@tool
def call_processing_agent(
    thinking_rationale: str,
    instruction: str,
    state: Annotated[SupervisorState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
):
    """
    Ask the Processing Agent to perform neuroimaging preprocessing with the appropriate tools, or to provide feedback to the Processing Agent during its execution.

    Inputs:
    - thinking_rationale: a concise thinking and reasoning rationale you have given the running history so far, so that the user can track the progress.
    - instruction: Directive or feedback for the Processing Agent.

    Returns:
    - The Processing Agent’s progress report to the Supervisor — the message the Processing Agent intends to communicate back to the Supervisor.

    ---
    Processing Agent can use some neuroimaging processing tools that are wrapped into python code to process sMRI and fMRI data. Specifically, the tools that Processing Agent can call and their brief descriptions are as follows:

    ## FSL Software: 
    - sMRI related tools:
    * fsl_bet_t1w: Perform brain extraction (skull stripping) on an sMRI image using FSL BET. There exists a parameter 'frac' can be adjusted: Fractional intensity threshold (default: 0.3). Higher values means more aggressive skull stripping (tend to remove more parts).
    * fsl_fast: Perform tissue segmentation on a skull-stripped structural image using FSL FAST. Prerequisites: This tool must be used after t1w skull stripping, but the skull stripping step can be performed by any tool or software (like AFNI, ANTs, etc.). It does not have to be FSL BET.
    * fsl_normalize_t1w_to_mni: Normalize T1w image to MNI space using FSL. Prerequisites: This tool must be used after t1w skull stripping, but the skull stripping step can be performed by any tool or software (like AFNI, ANTs, etc.).
    * fsl_warp_gm_tissue_seg_to_mni: Warp the Gray Matter Probability Map of the tissue segmentation to MNI space using FSL. Prerequisites: This tool must be used after 'fsl_normalize_t1w_to_mni' tool.

    - fMRI related tools:
    * fsl_slicetimer: Slice-timing correction for fMRI using FSL Slicetimer. There exists a parameter 'TR' can be adjusted: TR (time of repetition) in seconds. Default is 2.0.
    * fsl_motion_correct: Perform motion correction on a 4D fMRI image using FSL MCFLIRT.
    * fsl_coregister_func_to_anat: Co-register fMRI (EPI) to T1w structural image using FSL.
    * fsl_normalize_func_to_mni: Normalize fMRI to MNI space using FSL. Prerequisites: This function **must** be used after the `fsl_coregister_func_to_anat` function, which was used to co-register the fMRI to the T1w image.
    * fsl_spatial_smooth: Apply spatial smoothing to an image using FSL fslmaths. There exists a parameter 'fwhm' can be adjusted: Full-width at half maximum in millimeters (default: 2.0mm).
    * fsl_temporal_filter: Apply temporal band-pass filtering to a 4D fMRI image using FSL fslmaths -bptf. There exists parameters 'TR', 'highpass_freq', 'lowpass_freq' can be adjusted: Default is 2.0, 0.01, 0.1.


    ## AFNI Software:
    - sMRI related tools:
    * afni_t1w_skull_strip: Perform skull stripping on an sMRI image using AFNI's 3dSkullStrip.
    * afni_tissue_segmentation: Perform tissue segmentation on a skull-stripped sMRI image using AFNI's 3dSeg. Prerequisites: This tool must be used after t1w skull stripping, but the skull stripping step can be performed by any tool or software (like ANTs, FSL, etc.). It does not have to be AFNI 3dSkullStrip.
    * afni_normalize_t1w_to_mni: Normalize sMRI image to a standard mni template space using AFNI's @SSwarper.
    * afni_warp_gm_tissue_seg_to_mni: Warp the Gray Matter Probability Map of the tissue segmentation to MNI space using AFNI's 3dNwarpApply. Prerequisites: This tool must be used after 'afni_normalize_t1w_to_mni' tools.
    

    - fMRI related tools:
    * afni_fmri_slice_timing_correction: Perform slice-timing correction on fMRI data using AFNI's 3dTshift. There exists a parameter 'tpattern' can be adjusted: Slice acquisition pattern (e.g., alt+z, seq+z). Default is alt+z.
    * afni_fmri_motion_correction: Perform motion correction using AFNI's 3dvolreg.
    * afni_coregister_fmri_to_anat: Co-register fMRI to anatomical image using AFNI's (EPI to anat space).
    * afni_normalize_fmri_to_mni: Normalize fMRI to MNI space using AFNI's afni_proc.py. Use this tool, you do not need 'afni_coregister_fmri_to_anat' as prerequisite. This tool will internally perform the coregistration to the T1w image. Prerequisites: This tool must be used after the `afni_normalize_t1w_to_mni` tool, which was used to normalize the T1w image to MNI space with AFNI.
    * afni_spatial_smoothing: Apply spatial smoothing using AFNI's 3dmerge. There exists a parameter 'fwhm' can be adjusted: Full-width half maximum for Gaussian kernel (default: 2.0 mm).
    * afni_temporal_filter_fmri: Preprocess fMRI signal by regressing out motion, trends, and applying bandpass filtering using AFNI's 3dTproject. Prerequisites: This tool must be used after the `afni_fmri_motion_correction` tool. There exists parameters 'low_freq', 'high_freq', 'polort' can be adjusted: Default is 0.01, 0.1, 2.


    ## ANTs Software:
    - sMRI related tools:
    * ants_bias_field_correction: Perform N4 Bias Field Correction using ANTs N4BiasFieldCorrection. There exists a parameter 'shrink_factor' can be adjusted: Speed-up factor. Default: 4.
    * ants_skull_strip: Perform skull stripping using ANTs' antsBrainExtraction.sh script.
    * ants_tissue_segmentation: Perform tissue segmentation using ANTs Atropos. Prerequisites: This tool must be used after t1w skull stripping, but the skull stripping step can be performed by any tool or software (like AFNI, FSL, etc.).
    * ants_normalize_to_mni_template: Normalize t1w image to mni template space using ANTs antsRegistrationSyN. Prerequisites: This tool must be used after t1w skull stripping, but the skull stripping step can be performed by any tool or software (like AFNI, FSL, ANTs, etc.).
    * ants_warp_gm_tissue_seg_to_mni: Warp the Gray Matter Probability Map of the tissue segmentation to MNI space using ANTs. Prerequisites: This tool must be used after 'ants_normalize_to_mni_template' tool.

    ---

    **IMPORTANT RULES**:
    - When asking processing agent to process the data, you should tell it where the data is.
    - When assigning a task to the **processing agent**, you only need to specify **which software** to use (e.g., FSL, ANTs), **which parameters** to use, and **which data** need to be processed. You should **not** constrain it to specific commands or command-line instructions; the processing agent will decide on the appropriate commands based on your requirements.
    - When you are assigning tasks to the **processing agent**, you should assign tasks based on the available tools of the processing agent and within its capabilities.
    - You should determine the data processing pipeline based on the user’s actual requirements and the dependency relationships described in the tool specifications. Avoid unnecessary processing. Unless a tool description explicitly states a 'prerequisite', other tools can be considered independently usable (i.e., they can start directly from raw data).
    """



    # Prepare Processing Agent state from global state and the new instruction
    prior_proc_msgs = state.get("processing_agent_msgs", []) or []
    prior_sup_to_proc = state.get("supervisor_to_processing_instructions", []) or []
    prior_proc_to_sup = state.get("processing_to_supervisor_msgs", []) or []
    selected_tools = state.get("processing_agent_selected_tools", []) or []

    new_instruction_msg = HumanMessage(content=instruction)
    proc_input = {
        "selected_tools": selected_tools,
        "supervisor_instructions": prior_sup_to_proc + [new_instruction_msg],
        "processing_to_supervisor_msgs": prior_proc_to_sup,
        "messages": prior_proc_msgs,
    }


    # Invoke Processing Agent
    with get_openai_callback() as proc_cb:
        proc_out = ProcessingAgent.invoke(proc_input, config={"recursion_limit": 100})

    lastest_message_from_processing = proc_out.get("processing_to_supervisor_msgs")[-1].content

    # print(f"\nLastest message from processing: {lastest_message_from_processing}\n &&&&&&&&&\n")  # used for debugging


    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + ">>> BACKING TO MASTER CONTROL AGENT <<<".center(78) + "#")
    print("#" + " "*78 + "#")
    print("#"*80 + "\n")
    
    global global_sub_agent_tokens
    global_sub_agent_tokens["processing"]["total"] += proc_cb.total_tokens
    global_sub_agent_tokens["processing"]["prompt"] += proc_cb.prompt_tokens
    global_sub_agent_tokens["processing"]["completion"] += proc_cb.completion_tokens
    global_sub_agent_tokens["processing"]["cost"] += proc_cb.total_cost
    
    print(f"[Token Usage] Processing Agent consumed: {proc_cb.total_tokens} tokens (Prompt: {proc_cb.prompt_tokens}, Completion: {proc_cb.completion_tokens}) - Cost: ${proc_cb.total_cost:.4f}")


    return Command(
        update={
            "messages": [
                ToolMessage(lastest_message_from_processing, tool_call_id=tool_call_id)
                ],
            "processing_agent_msgs": proc_out.get("messages", prior_proc_msgs),
            "processing_agent_selected_tools": proc_out.get("selected_tools", selected_tools),
            "supervisor_to_processing_instructions": prior_sup_to_proc + [new_instruction_msg],
            "processing_to_supervisor_msgs": proc_out.get("processing_to_supervisor_msgs"),
        }
    )


@tool
def call_quality_control_agent(
    thinking_rationale: str,
    instruction: str,
    state: Annotated[SupervisorState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
):
    """
    Ask the Quality Control (QC) Agent to perform quality control on the data using the appropriate tools. QC Agent can tell you binary verdict (pass or fail) for the data quality, no borderline cases considered. 

    Inputs:
    - thinking_rationale: a concise thinking and reasoning rationale you have given the running history so far, so that the user can track the progress.
    - instruction: Directive you want the Quality Control Agent to follow or feedback for the Quality Control Agent.

    Returns:
    - The Quality Control Agent’s evaluation of the data quality or feedback to the Supervisor.

    ---

    The Quality Control Agent is responsible for evaluating MRI data quality at different stages of the workflow.

    1. Quality Control on Raw Data
    The Quality Control Agent may use MRIQC and related tools to assess the quality of raw, unprocessed MRI data.
    Based on the Binary (pass or fail) evaluation results, the agent should determine which subjects have insufficient data quality and should be excluded from subsequent analysis. No borderline cases considered.

    2. Quality Control on Preprocessed Data
    After the Processing Agent completes preprocessing (for sampled subjects or for all subjects), the Quality Control Agent need to perform quality control on the processed data.

    For preprocessed data, the Quality Control Agent may perform the some of the following checks:

    - Quality control of the skull stripping step for structural MRI.
    - Quality control of the tissue segmentation step for structural MRI.
    - Quality control of the T1w-to-MNI normalization step.
    - Quality control of the fMRI-to-anatomical co-registration step.
    - Quality control of the fMRI normalization-to-MNI step.

    If, after a given processing pipeline, sampled subjects fail quality control, you—as the supervisor—should adjust the processing pipeline based on the Quality Control Agent’s feedback. This may involve changing processing tools or modifying parameters (if applicable).

    # IMPORTANT RULES:
    - When instructing the Quality Control Agent to perform quality control, you must explicitly specify:
        1. The location of the raw (unprocessed) data.
            - Even when performing QC on processed data, you must still provide the path to the corresponding raw data.
        2. The location of the processed data that requires QC.
            - Do not provide only the top-level directory.
            - You must specify the exact file paths and the consistent naming pattern used.
            - Do not list subjects one by one. Instead, describe the unified file path structure or naming convention that applies to all relevant subjects.
        3. Which subjects require QC.
    The Quality Control Agent must have sufficient and unambiguous file path information to locate the data without making assumptions.
    
    - You are the one who decide how to reprocess the subjects who failed quality control. You should not ask the Quality Control Agent to give you reprocessing advice.
    - You must determine which quality control (QC) checks to perform based on:
        - The current processing stage,
        - The user’s final task requirements, and
        - The capabilities of the Quality Control Agent.

    - Only perform QC checks that are relevant to the user’s requested task. Do not perform QC checks that are unrelated to the user’s request.
    - When the Quality Control sub-agent performs QC, they need the exact locations of the output files produced by the Processing sub-agent, as well as the file naming conventions (not just a high-level folder path). When you assign tasks to the Quality Control sub-agent, you must provide complete and specific file location details. If you are not sure, you must confirm with the Processing Agent first.
    """

    # Prepare Quality Control Agent state from global state and the new instruction
    prior_qc_msgs = state.get("quality_control_agent_msgs", []) or []
    prior_sup_to_qc = state.get("supervisor_to_quality_control_instructions", []) or []
    prior_qc_to_sup = state.get("quality_control_to_supervisor_msgs", []) or []
    selected_tools = state.get("quality_control_agent_selected_tools", []) or []
    selected_knowledge_docs = state.get("quality_control_agent_selected_knowledge_docs", []) or []


    new_instruction_msg = HumanMessage(content=instruction)
    quality_control_input = {
        "selected_tools": selected_tools,
        "supervisor_instructions": prior_sup_to_qc + [new_instruction_msg],
        "quality_control_to_supervisor_msgs": prior_qc_to_sup,
        "selected_knowledge_docs": selected_knowledge_docs,
        "messages": prior_qc_msgs,
    }

    # Invoke Quality Control Agent
    with get_openai_callback() as qc_cb:
        quality_control_out = QualityControlAgent.invoke(quality_control_input, config={"recursion_limit": 100})

    lastest_message_from_quality_control = quality_control_out.get("quality_control_to_supervisor_msgs")[-1].content

    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + ">>> BACKING TO MASTER CONTROL AGENT <<<".center(78) + "#")
    print("#" + " "*78 + "#")
    print("#"*80 + "\n")
    
    global global_sub_agent_tokens
    global_sub_agent_tokens["quality_control"]["total"] += qc_cb.total_tokens
    global_sub_agent_tokens["quality_control"]["prompt"] += qc_cb.prompt_tokens
    global_sub_agent_tokens["quality_control"]["completion"] += qc_cb.completion_tokens
    global_sub_agent_tokens["quality_control"]["cost"] += qc_cb.total_cost
    
    print(f"[Token Usage] Quality Control Agent consumed: {qc_cb.total_tokens} tokens (Prompt: {qc_cb.prompt_tokens}, Completion: {qc_cb.completion_tokens}) - Cost: ${qc_cb.total_cost:.4f}")


    return Command(
        update={
            "messages": [
                ToolMessage(lastest_message_from_quality_control, tool_call_id=tool_call_id)
                ],
            "quality_control_agent_msgs": quality_control_out.get("messages", prior_qc_msgs),
            "quality_control_agent_selected_tools": quality_control_out.get("selected_tools", selected_tools),
            "quality_control_agent_selected_knowledge_docs": quality_control_out.get("selected_knowledge_docs", selected_knowledge_docs),
            "supervisor_to_quality_control_instructions": prior_sup_to_qc + [new_instruction_msg],
            "quality_control_to_supervisor_msgs": quality_control_out.get("quality_control_to_supervisor_msgs"),
        }
    )



@tool
def call_data_awareness_agent(
    thinking_rationale: str,
    instruction: str,
    state: Annotated[SupervisorState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
):
    """
    Ask the Data Awareness sub-Agent to get the information of the data. Such as the meta-data of neuroimaging data, specific data directory structure, etc.

    Inputs:
    - thinking_rationale: a concise thinking and reasoning rationale you have given the running history so far, so that the user can track the progress.
    - instruction: Tell the Data Awareness sub-Agent what information you want to get about the data.

    Returns:
    - The reply from the Data Awareness sub-Agent.

    ---

    """

    # Prepare Data Awareness sub-Agent state from global state and the new instruction
    prior_da_msgs = state.get("data_awareness_agent_msgs", []) or []
    prior_sup_to_da = state.get("supervisor_to_data_awareness_instructions", []) or []
    prior_da_to_sup = state.get("data_awareness_to_supervisor_msgs", []) or []
    selected_tools = state.get("data_awareness_agent_selected_tools", []) or []

    new_instruction_msg = HumanMessage(content=instruction)
    data_awareness_input = {
        "selected_tools": selected_tools,
        "supervisor_instructions": prior_sup_to_da + [new_instruction_msg],
        "data_awareness_to_supervisor_msgs": prior_da_to_sup,
        "messages": prior_da_msgs,
    }

    # Invoke Data Awareness sub-Agent
    with get_openai_callback() as da_cb:
        data_awareness_out = DataAwarenessAgent.invoke(data_awareness_input, config={"recursion_limit": 100})

    lastest_message_from_data_awareness = data_awareness_out.get("data_awareness_to_supervisor_msgs")[-1].content

    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + ">>> BACKING TO MASTER CONTROL AGENT <<<".center(78) + "#")
    print("#" + " "*78 + "#")
    print("#"*80 + "\n")
    
    global global_sub_agent_tokens
    global_sub_agent_tokens["data_awareness"]["total"] += da_cb.total_tokens
    global_sub_agent_tokens["data_awareness"]["prompt"] += da_cb.prompt_tokens
    global_sub_agent_tokens["data_awareness"]["completion"] += da_cb.completion_tokens
    global_sub_agent_tokens["data_awareness"]["cost"] += da_cb.total_cost
    
    print(f"[Token Usage] Data Awareness Agent consumed: {da_cb.total_tokens} tokens (Prompt: {da_cb.prompt_tokens}, Completion: {da_cb.completion_tokens}) - Cost: ${da_cb.total_cost:.4f}")


    return Command(
        update={
            "messages": [
                ToolMessage(lastest_message_from_data_awareness, tool_call_id=tool_call_id)
                ],
            "data_awareness_agent_msgs": data_awareness_out.get("messages", prior_da_msgs),
            "data_awareness_agent_selected_tools": data_awareness_out.get("selected_tools", selected_tools),
            "supervisor_to_data_awareness_instructions": prior_sup_to_da + [new_instruction_msg],
            "data_awareness_to_supervisor_msgs": data_awareness_out.get("data_awareness_to_supervisor_msgs"),
        }
    )


@tool
def call_downstream_analysis_agent(
    thinking_rationale: str,
    instruction: str,
    state: Annotated[SupervisorState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
):
    """
    Ask the Downstream Analysis sub-Agent to perform downstream analysis tasks (e.g., prediction modeling).

    Inputs:
    - thinking_rationale: a concise thinking and reasoning rationale you have given the running history so far, so that the user can track the progress.
    - instruction: Tell the Downstream Analysis sub-Agent what analysis task you want it to perform and any feedback you want to provide to the Downstream Analysis sub-Agent.

    Returns:
    - The feedback from the Downstream Analysis sub-Agent.

    ---

    Currently, the Downstream Analysis sub-Agent can perform the following analysis tasks:
    - Based on the preprocessed fMRI or sMRI data of subjects, construct and train model(s) to predict certain indicators, such as classification indicators.
    - Other downstream analysis tasks based on the preprocessed MRI data.

    ---

    Note that:
    - When asking downstream analysis agent to perform the analysis task, you should clearly inform it where the data needed for the analysis task is and which subjects’ data should be used for the analysis (or which subjects’ data should be excluded from the analysis due to quality control issues). Do not simply provide a folder location, as there may be many other unrelated files within the folder.
    """

    # Prepare Downstream Analysis sub-Agent state from global state and the new instruction
    prior_da_msgs = state.get("downstream_analysis_agent_msgs", []) or []
    prior_sup_to_da = state.get("supervisor_to_downstream_analysis_instructions", []) or []
    prior_da_to_sup = state.get("downstream_analysis_to_supervisor_msgs", []) or []
    selected_tools = state.get("downstream_analysis_agent_selected_tools", []) or []
    selected_knowledge_docs = state.get("downstream_analysis_agent_selected_knowledge_docs", []) or []

    new_instruction_msg = HumanMessage(content=instruction)
    downstream_analysis_input = {
        "selected_tools": selected_tools,
        "selected_knowledge_docs": selected_knowledge_docs,
        "supervisor_instructions": prior_sup_to_da + [new_instruction_msg],
        "downstream_analysis_to_supervisor_msgs": prior_da_to_sup,
        "messages": prior_da_msgs,
    }

    # Invoke Downstream Analysis sub-Agent
    with get_openai_callback() as ds_cb:
        downstream_analysis_out = DownstreamAnalysisAgent.invoke(downstream_analysis_input, config={"recursion_limit": 100})

    lastest_message_from_downstream_analysis = downstream_analysis_out.get("downstream_analysis_to_supervisor_msgs")[-1].content

    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + ">>> BACKING TO MASTER CONTROL AGENT <<<".center(78) + "#")
    print("#" + " "*78 + "#")
    print("#"*80 + "\n")
    
    global global_sub_agent_tokens
    global_sub_agent_tokens["downstream_analysis"]["total"] += ds_cb.total_tokens
    global_sub_agent_tokens["downstream_analysis"]["prompt"] += ds_cb.prompt_tokens
    global_sub_agent_tokens["downstream_analysis"]["completion"] += ds_cb.completion_tokens
    global_sub_agent_tokens["downstream_analysis"]["cost"] += ds_cb.total_cost
    
    print(f"[Token Usage] Downstream Analysis Agent consumed: {ds_cb.total_tokens} tokens (Prompt: {ds_cb.prompt_tokens}, Completion: {ds_cb.completion_tokens}) - Cost: ${ds_cb.total_cost:.4f}")


    return Command(
        update={
            "messages": [
                ToolMessage(lastest_message_from_downstream_analysis, tool_call_id=tool_call_id)
                ],
            "downstream_analysis_agent_msgs": downstream_analysis_out.get("messages", prior_da_msgs),
            "downstream_analysis_agent_selected_tools": downstream_analysis_out.get("selected_tools", selected_tools),
            "downstream_analysis_agent_selected_knowledge_docs": downstream_analysis_out.get("selected_knowledge_docs", selected_knowledge_docs),
            "supervisor_to_downstream_analysis_instructions": prior_sup_to_da + [new_instruction_msg],
            "downstream_analysis_to_supervisor_msgs": downstream_analysis_out.get("downstream_analysis_to_supervisor_msgs"),
        }
    )



SUPERVISOR_SYSTEM_PROMPT = """
You are an **advanced neuroscientist**. You understand the user’s needs and assign the corresponding tasks to professionals with the required expertise.

You act as an **overall planner and supervisor**, collaborating with skilled professionals (sub-agents) to accomplish the user’s tasks.

Currently, the professionals working with you include:
- the **Quality Control sub-Agent**, who can perform quality control on the data using the appropriate tools and report whether the data passes quality control.
- the **Processing sub-Agent**, who can carry out MRI data preprocessing tasks according to your instructions.
- the **Data Awareness sub-Agent**, who can get the information of the data you need. Such as the meta-data of neuroimaging data, specific data directory structure, etc.
- the **Downstream Analysis sub-Agent**, who can perform downstream analysis tasks (e.g., prediction modeling based on the processed data).


You are the “brain” and overall commander of the entire system. You are responsible for task decomposition, strategic planning, team coordination, and making dynamic decisions based on run-time feedback. Other specialized agents rely on your coordination and direction—you must clearly instruct them on what to do and how to do it.

During this collaboration, you communicate with these professionals and supervise their work to ensure they complete the tasks according to your requirements. While performing the tasks, the staff will report their progress to you or seek your assistance.
Note that all professionals can only communicate directly with you; they do not communicate with each other. You are the only bridge between them.

**Important**: Every task you assign to a sub-agent must fall within its capabilities. If you are unsure about what a sub-agent can do—such as what types of analyses it can perform or which tools it can use—you must ask the sub-agent first rather than making assumptions. Assigning tasks beyond a sub-agent’s capabilities is not allowed.


Important Rules:
- On the premise of ensuring information integrity and accuracy, your communication with the professional sub-agents should be concise and clear.
- You can only call **one sub-agent at a time** within a single conversation turn.
- The only Workspace is 'Path/temp_workspace', all files, intermediate results, derivatives, etc. should be stored in this folder, and not in any other place.
- If you need certain detailed information in decision-making but are unsure, do not make assumptions. Ask the sub-agent for clarification before making a decision.
- When the Quality Control sub-agent performs QC, it needs the exact locations of the output files produced by the Processing sub-agent, as well as the file naming conventions (not just a high-level folder path). When assigning tasks to the Quality Control sub-agent, you must provide complete and specific file location details. If the Processing Agent tells you what data was produced and where it is located, you must relay that information to the QC Agent in full and without omission.
- If the QC Agent lacks the necessary information, they may ask you to confirm it. In that case, you should provide the missing details. The Processing Agent likely already gave you the required file information earlier—you just didn’t pass it along to the QC Agent.
- If you’re not sure, you should confirm with the Processing Agent. When checking with the Processing Agent, you can ask them to verify the exact locations and naming conventions and have them tell you the precise details directly, rather than asking them to redo the processing work.
"""


llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"), 
    model="gpt-5.2"
)


supervisor_agent = create_react_agent(
    model=llm,
    tools=[call_processing_agent, call_quality_control_agent, call_data_awareness_agent, call_downstream_analysis_agent],
    prompt=SUPERVISOR_SYSTEM_PROMPT,
    state_schema=SupervisorState
)


if __name__ == "__main__":

    import time


######### +++++++++++++++++++++++++++++++++++++++++ ADHD 200 task +++++++++++++++++++++++++++++++++++++++++ #########
#     user_request = """
# In `Path/ADHD200_RawData`, there are raw data from several ADHD200 subjects. Your task is to preprocess these subjects’ data, then train classification model(s) for functional brain connectome analysis based on the preprocessed data.

# The target label is **diagnosis** (Control vs ADHD), which can be found in:
# `Path/ADHD200_RawData/participants.tsv`.

# Your final deliverables are:

# 1. A complete neuroimaging preprocessing pipeline.
# 2. Trained downstream prediction model.
# 3. The corresponding inference script that can load the trained model(s) and produce predictions on the held-out test set.

# The provided dataset should be treated as training set. Your delivered preprocessing pipeline and model will be applied to another held-out test set of subjects (which is invisible to you). Your performance will be evaluated based on the prediction score on this test set. 
# This is a competitive evaluation. The primary objective is to achieve the highest possible score on the held-out test set, while ensuring that the workflow remains methodologically sound, logically justified, reproducible, and free from data leakage or invalid assumptions.

# You can explore different preprocessing pipelines (**up to three**) using available neuroimaging tools and design different modeling methods to find the one that potentially yields the best prediction performance on the held-out test set.

# Besides the final deliverables, you should also clearly explain the rationale behind the final selection of the preprocessing pipeline and the corresponding modeling methods.
#     """


    user_request = """
In `Path/ADHD200_RawData_Sampled`, there are raw data from several ADHD200 subjects. Your task is to preprocess these subjects’ data, then train classification model(s) for functional brain connectome analysis based on the preprocessed data.

The target label is **diagnosis** (Control vs ADHD), which can be found in:
`Path/ADHD200_RawData_Sampled/participants.tsv`.

Your final deliverables are:

1. A complete neuroimaging preprocessing pipeline.
2. Trained downstream prediction model.
3. The corresponding inference script that can load the trained model(s) and produce predictions on the held-out test set.

The provided dataset should be treated as training set. Your delivered preprocessing pipeline and model will be applied to another held-out test set of subjects (which is invisible to you). Your performance will be evaluated based on the prediction score on this test set. 
This is a competitive evaluation. The primary objective is to achieve the highest possible score on the held-out test set, while ensuring that the workflow remains methodologically sound, logically justified, reproducible, and free from data leakage or invalid assumptions.

Besides the final deliverables, you should also clearly explain the rationale behind the final selection of the preprocessing pipeline and the corresponding modeling methods.
    """



######### =================================== ADNI task =================================== #########

#     user_request = """
# In `Path/ADNI_RawData`, there are raw sMRI data from several ADNI subjects. Your task is to preprocess these subjects’ data, then train classification model(s) using voxel-wise gray matter probability map features derived from voxel-based morphometry (VBM).

# The target label is **diagnosis** (CN vs AD), which can be found in:
# `Path/ADNI_RawData/participants.tsv`.

# Your final deliverables are:

# 1. A complete neuroimaging preprocessing pipeline.
# 2. Trained downstream prediction model.
# 3. The corresponding inference script that can load the trained model and produce predictions on the held-out test set.

# The provided dataset should be treated as training set. Your delivered preprocessing pipeline and model will be applied to another held-out test set of subjects (which is invisible to you). Your performance will be evaluated based on the prediction score on this test set. This is a competitive evaluation — achieving the highest possible score in the held-out test set is the primary objective of this task. Note that you have only one final submission opportunity for this task. You must not assume that you can submit once, observe the outcome, and then revise or resubmit later.

# You can explore different preprocessing pipelines (**up to three**) using available neuroimaging tools and design different modeling methods to find the one that potentially yields the best prediction performance on the held-out test set.

# Besides the final deliverables, you should also clearly explain the rationale behind the final selection of the preprocessing pipeline and the corresponding modeling methods.
#     """


    input_state = {
        "messages": [HumanMessage(content=user_request)],
        "user_request": user_request,

        # Processing Agent State
        "processing_agent_msgs": [],
        "processing_agent_selected_tools": [],
        "supervisor_to_processing_instructions": [],
        "processing_to_supervisor_msgs": [],

        # Quality Control Agent State
        "quality_control_agent_msgs": [],
        "quality_control_agent_selected_tools": [],
        "quality_control_agent_selected_knowledge_docs": [],
        "supervisor_to_quality_control_instructions": [],
        "quality_control_to_supervisor_msgs": [],

        # Data Awareness Agent State
        "data_awareness_agent_msgs": [],
        "data_awareness_agent_selected_tools": [],
        "supervisor_to_data_awareness_instructions": [],
        "data_awareness_to_supervisor_msgs": [],

        # Downstream Analysis Agent State
        "downstream_analysis_agent_msgs": [],
        "downstream_analysis_agent_selected_tools": [],
        "downstream_analysis_agent_selected_knowledge_docs": [],
        "supervisor_to_downstream_analysis_instructions": [],
        "downstream_analysis_to_supervisor_msgs": [],
    }


    overall_start_time = time.time()
    start_time = time.time()

    final_state = input_state
    with get_openai_callback() as total_cb:
        for step in supervisor_agent.stream(input_state, stream_mode="values", config={"recursion_limit": 100}):
            final_state = step
            last_message = step["messages"][-1]

            print('\n' +'*'*5 + '[Master Control Agent]' + '*'*5)
            last_message.pretty_print()

            print(f'this step took {time.time() - start_time} seconds')
            start_time = time.time()

            print(f'till now, time taken: {time.time() - overall_start_time} seconds')

            # used for debugging
            # print('#'*30)
            # print(step)
            # print('#'*30 + '\n\n')

    end_time = time.time()
    
    print("\n" + "*"*50 + "\n")
    print(f"ALL Time taken: {end_time - overall_start_time} seconds")
    print("\n" + "="*50)
    print("TOKEN USAGE SUMMARY")
    print("="*50)
    
    # Calculate Supervisor's own token usage
    total_sub_agent_tokens = sum(agent["total"] for agent in global_sub_agent_tokens.values())
    total_sub_agent_prompt = sum(agent["prompt"] for agent in global_sub_agent_tokens.values())
    total_sub_agent_completion = sum(agent["completion"] for agent in global_sub_agent_tokens.values())
    total_sub_agent_cost = sum(agent["cost"] for agent in global_sub_agent_tokens.values())
    
    supervisor_total = total_cb.total_tokens - total_sub_agent_tokens
    supervisor_prompt = total_cb.prompt_tokens - total_sub_agent_prompt
    supervisor_completion = total_cb.completion_tokens - total_sub_agent_completion
    supervisor_cost = total_cb.total_cost - total_sub_agent_cost

    print("\n[Supervisor Agent (Master Control)]")
    print(f"  Total Tokens: {supervisor_total} (Prompt: {supervisor_prompt}, Completion: {supervisor_completion})")
    print(f"  Cost: ${supervisor_cost:.4f}")
    
    print("\n[Sub-Agents]")
    for agent_name, usage in global_sub_agent_tokens.items():
        if usage["total"] > 0:
            print(f"  - {agent_name.replace('_', ' ').title()}:")
            print(f"      Total Tokens: {usage['total']} (Prompt: {usage['prompt']}, Completion: {usage['completion']})")
            print(f"      Cost: ${usage['cost']:.4f}")

    print("\n[OVERALL SYSTEM]")
    print(f"  Total Tokens: {total_cb.total_tokens} (Prompt: {total_cb.prompt_tokens}, Completion: {total_cb.completion_tokens})")
    print(f"  Total Cost: ${total_cb.total_cost:.4f}")
    print("="*50 + "\n")

    print("\n" + "=" * 50)
    print("PER-AGENT FINAL MESSAGES TOKEN COUNT")
    print("=" * 50)

    agent_message_sources = [
        ("Supervisor (Master Control)", final_state.get("messages", []) or []),
        ("Processing Agent", final_state.get("processing_agent_msgs", []) or []),
        ("Quality Control Agent", final_state.get("quality_control_agent_msgs", []) or []),
        ("Data Awareness Agent", final_state.get("data_awareness_agent_msgs", []) or []),
        ("Downstream Analysis Agent", final_state.get("downstream_analysis_agent_msgs", []) or []),
    ]

    grand_total_messages_tokens = 0
    for agent_name, agent_messages in agent_message_sources:
        messages_tokens = _count_messages_tokens_safe(llm, agent_messages)
        grand_total_messages_tokens += messages_tokens
        print(
            f"  - {agent_name}: "
            f"messages_count={len(agent_messages)} | messages_tokens={messages_tokens}"
        )

    print(f"\n  [SUM over all agents' messages] tokens = {grand_total_messages_tokens}")
    print("=" * 50 + "\n")

