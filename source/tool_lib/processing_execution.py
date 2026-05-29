from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.types import Command
import subprocess
from dotenv import load_dotenv
import os
import time



# write todo tool
# https://github.com/langchain-ai/langchain/blob/c63f23d2339b2604edc9ae1d9f7faf7d6cc7dc78/libs/langchain_v1/langchain/agents/middleware/todo.py#L121

WRITE_TODOS_TOOL_DESCRIPTION = """Use this tool to create and manage a structured task list for your current work session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.

Only use this tool if you think it will be helpful in staying organized. If the user's request is trivial and takes less than 3 steps, it is better to NOT use this tool and just do the task directly.

## When to Use This Tool
Use this tool in these scenarios:

1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions
2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations
3. User explicitly requests todo list - When the user directly asks you to use the todo list
4. User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)
5. The plan may need future revisions or updates based on results from the first few steps

## How to Use This Tool
1. When you start working on a task - Mark it as in_progress BEFORE beginning work.
2. After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation.
3. You can also update future tasks, such as deleting them if they are no longer necessary, or adding new tasks that are necessary. Don't change previously completed tasks.
4. You can make several updates to the todo list at once. For example, when you complete a task, you can mark the next task you need to start as in_progress.

## When NOT to Use This Tool
It is important to skip using this tool when:
1. There is only a single, straightforward task
2. The task is trivial and tracking it provides no benefit
3. The task can be completed in less than 3 trivial steps
4. The task is purely conversational or informational

## Task States and Management

1. **Task States**: Use these states to track progress:
   - pending: Task not yet started
   - in_progress: Currently working on (you can have multiple tasks in_progress at a time if they are not related to each other and can be run in parallel)
   - completed: Task finished successfully

2. **Task Management**:
   - Update task status in real-time as you work
   - Mark tasks complete IMMEDIATELY after finishing (don't batch completions)
   - Complete current tasks before starting new ones
   - Remove tasks that are no longer relevant from the list entirely
   - IMPORTANT: When you write this todo list, you should mark your first task (or tasks) as in_progress immediately!.
   - IMPORTANT: Unless all tasks are completed, you should always have at least one task in_progress to show the user that you are working on something.

3. **Task Completion Requirements**:
   - ONLY mark a task as completed when you have FULLY accomplished it
   - If you encounter errors, blockers, or cannot finish, keep the task as in_progress
   - When blocked, create a new task describing what needs to be resolved
   - Never mark a task as completed if:
     - There are unresolved issues or errors
     - Work is partial or incomplete
     - You encountered blockers that prevent completion
     - You couldn't find necessary resources or dependencies
     - Quality standards haven't been met

4. **Task Breakdown**:
   - Create specific, actionable items
   - Break complex tasks into smaller, manageable steps
   - Use clear, descriptive task names

Being proactive with task management demonstrates attentiveness and ensures you complete all requirements successfully
Remember: If you only need to make a few tool calls to complete a task, and it is clear what you need to do, it is better to just do the task directly and NOT call this tool at all."""

class Todo(TypedDict):
    """A single todo item with content and status."""

    content: str
    """The content/description of the todo item."""

    status: Literal["pending", "in_progress", "completed"]
    """The current status of the todo item."""


@tool(description=WRITE_TODOS_TOOL_DESCRIPTION)
def write_todos(todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]):
    """Create and manage a structured task list for your current work session."""
    return Command(
        update={
            "todos": todos,
            "messages": [ToolMessage(f"Updated todo list successfully.", tool_call_id=tool_call_id)],
        }
    )



from pathlib import Path

def find_project_root(start: Path | None = None) -> Path:
    if start is None:
        start = Path(__file__).resolve()

    for parent in [start, *start.parents]:
        tool_pkg = parent / "tool_lib" / "__init__.py"
        if tool_pkg.is_file():
            return parent


# run terminal cmd tool

def get_subprocess_env():
    load_dotenv()

    PROJECT_ROOT = find_project_root()

    env = os.environ.copy()
    system_path = env["PATH"]

    python_dir = env.get("PYTHON_DIR", "")
    ants_dir   = env.get("ANTS_DIR", "")
    afni_dir   = env.get("AFNI_DIR", "")
    fsldir     = env.get("FSLDIR", "")

    paths = []
    if python_dir: 
        paths.append(python_dir)
    if ants_dir:
        paths.append(ants_dir)
    if afni_dir:
        paths.append(afni_dir)
    if fsldir:
        paths.append(os.path.join(fsldir, "bin"))

    paths.append(system_path)

    new_path = ":".join(paths)
    env["PATH"] = new_path

    existing_pp = env.get("PYTHONPATH", "")
    new_pp_parts = [str(PROJECT_ROOT)]
    if existing_pp:
        new_pp_parts.append(existing_pp)
    env["PYTHONPATH"] = os.pathsep.join(new_pp_parts)


    return env


def truncate_head_tail(text: str, head_len: int = 2000, tail_len: int = 2000) -> str:
    if len(text) <= head_len + tail_len:
        return text, False
    return (
        text[:head_len]
        + "\n[...TRUNCATED...]\n"
        + text[-tail_len:]
    ), True


@tool
def run_bash_command(command: str) -> str:
    """
    Execute the given string as a bash command. 

    Pass the actual commands/script directly; DO NOT wrap it again as bash -lc "...".
    """

    import re

    def ensure_sbatch_wait(command: str) -> str:
        sbatch_pattern = r'(?<!\S)sbatch(?!\S)'

        if re.search(sbatch_pattern, command):
            if "--wait" not in command:
                command = re.sub(
                    sbatch_pattern,
                    "sbatch --wait",
                    command,
                    count=1
                )
        return command



    # Basic safeguard: block obviously dangerous shell commands
    lowered = command.lower()
    forbidden_substrings = [
        " rm -rf",
        " rm -r",
        " rm --recursive",
        " rm /",
        " rm ",
        "rm -rf /",
        "rm -r /",
        "sudo rm",
        " mkfs",
        " fdisk",
        " shutdown",
        " reboot",
        ":(){:|:&};:",  # fork bomb
    ]

    for pattern in forbidden_substrings:
        if pattern in lowered:
            return (
                "Security policy: detected dangerous command, execution blocked.\n"
                f"Blocked command: {command}\n"
            )

    env = get_subprocess_env()

    # user_input = input("Proceed with command execution? \nCommand: {command}\n(y/n): ").lower()


    LOG_DIR = "Path/temp_workspace/temp_log"
    os.makedirs(LOG_DIR, exist_ok=True)

    # Create a timestamped log file name
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile_path = os.path.join(LOG_DIR, f"cmd_{timestamp}.log")

    user_input = "y" # default to yes
    if user_input == "y":
        try:
            command = ensure_sbatch_wait(command)

            result = subprocess.run(command, shell=True, capture_output=True, text=True, env=env, executable="/bin/bash")

            if "sbatch" not in command:
                # Write full stdout/stderr to logfile
                with open(logfile_path, "w") as f:
                    f.write(f"Command executed: {command}\n\n")
                    f.write(f"=== STDOUT ===\n{result.stdout}\n\n")
                    f.write(f"=== STDERR ===\n{result.stderr}\n\n")
                    f.write(f"Return code: {result.returncode}\n")

                # Summarize or truncate
                truncated_stdout, is_truncated_stdout = truncate_head_tail(result.stdout)
                truncated_stderr, is_truncated_stderr = truncate_head_tail(result.stderr)

                if is_truncated_stdout or is_truncated_stderr:
                    summary = (
                        f"Command Return Code: {result.returncode}\n\n"
                        f"The output of the command is truncated because it is too long. If you need the full output, you can read the full output in the log file."
                        f"STDOUT:\n{truncated_stdout}\n\n"
                        f"STDERR:\n{truncated_stderr}\n\n"
                        f"Full logs saved to: {logfile_path}"
                    )
                else:
                    summary = (
                        f"Command Return Code: {result.returncode}\n\n"
                        f"STDOUT:\n{result.stdout}\n\n"
                        f"STDERR:\n{result.stderr}\n\n"
                    )

                return summary
            else:
                return "The task submitted by sbatch has completed, If needed, you should read the log file defined in sbatch script `.sh` file for the log."

          

        except Exception as e:
            # If there's a more fundamental error (e.g. Python environment, permissions)
            return f"Failed to run command: {command}\n\nError: {str(e)}"
    else:
        return "Invalid input. User aborted command execution."





# tools for writing files
@tool
def write_file(file_path: str, content: str) -> str:
    """Write text content to a file at absolute path. Overwrites existing content.

    Args:
        file_path: Absolute path of the file to write.
        content: Text content to write into the file.

    Returns:
        A short status message describing the write result.
    """
    if not os.path.isabs(file_path):
        raise ValueError(f"`file_path` must be an absolute path. Got: {file_path!r}")

    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    return f"File written successfully to '{file_path}'."



@tool
def read_file(
    file_path: str,
    offset: int = 0,
    line_count: int = 20,
    max_chars: int = 1000
) -> str:
    """
    Read text content from a file at absolute path. Returns the file content as a string.

    Args:
    - file_path: Absolute path of the file to read.
    - offset: the number of lines to skip from the beginning of the file. 0 means start from the beginning of the file. -1 means read from the tail of the file.
    - line_count: the number of lines to read from the file. You should start with a small value, such as 20.
    - max_chars: Maximum number of characters to return (default: 1000).

    Returns:
    - A string of the read content.
    
    Note:
    - Always be mindful of potential token explosion. You can read at most 4000 tokens at a time. If you need to read more, you can read the file in chunks.
    - For log files, it is better to read the tail of the file.

    """

    if max_chars > 4000:
        max_chars = 4000

    if not os.path.isabs(file_path):
        raise ValueError(f"`file_path` must be an absolute path. Got: {file_path!r}")

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: '{file_path}'")

    if max_chars is not None and max_chars < 0:
        raise ValueError(f"`max_chars` must be >= 0 or None. Got: {max_chars}")

    MARK_AFTER = "[Some contents are truncated after this point]"
    MARK_BEFORE = "[Some contents are truncated before this point]"

    def _ensure_trailing_newline(s: str) -> str:
        return s if (s == "" or s.endswith("\n")) else (s + "\n")

    def _append_notice(rendered: str, *, original_chars: int, kept_content: str, tail_mode: bool) -> str:
        # Stats must NOT include the marker (and naturally also not include the notice itself).
        truncated_chars = len(kept_content)
        truncated_lines = len(kept_content.splitlines())

        direction = " Tail-mode truncation was applied from the end backwards." if tail_mode else ""
        notice = (
            "NOTICE: Content was too long and has been truncated."
            f"{direction} "
            f"Original chars: {original_chars}, "
            f"After Truncated chars: {truncated_chars}, "
            f"After Truncated lines: {truncated_lines}"
        )
        return _ensure_trailing_newline(rendered) + notice

    # ---------------------------
    # Tail mode: read last lines
    # ---------------------------
    if offset == -1:
        if line_count <= 0:
            return ""

        try:
            result = subprocess.run(
                ["tail", "-n", str(line_count), file_path],
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to run 'tail' on '{file_path}': {e}") from e

        if result.returncode != 0:
            raise RuntimeError(
                f"'tail' command failed for '{file_path}' with code {result.returncode}: "
                f"{result.stderr.strip()}"
            )

        content = result.stdout
        original_chars = len(content)

        if max_chars is None or original_chars <= max_chars:
            return content

        kept = content[-max_chars:] if max_chars > 0 else ""
        rendered = _ensure_trailing_newline(MARK_BEFORE) + kept
        return _append_notice(rendered, original_chars=original_chars, kept_content=kept, tail_mode=True)

    # ---------------------------
    # Head mode: read forward
    # ---------------------------
    if offset < 0:
        raise ValueError(f"`offset` must be >= 0 or -1 for tail mode. Got: {offset}")

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        # Skip `offset` lines.
        for _ in range(offset):
            line = f.readline()
            if line == "":
                return ""

        original_chars = 0
        kept_parts: list[str] = []
        kept_len = 0

        def _keep_head(s: str) -> None:
            nonlocal kept_len
            if max_chars is None:
                kept_parts.append(s)
                kept_len += len(s)
                return
            remaining = max_chars - kept_len
            if remaining <= 0:
                return
            kept_parts.append(s[:remaining])
            kept_len += min(remaining, len(s))

        # If line_count <= 0, read until EOF (streaming).
        if line_count <= 0:
            while True:
                chunk = f.read(8192)
                if chunk == "":
                    break
                original_chars += len(chunk)
                _keep_head(chunk)

            kept = "".join(kept_parts)
            if max_chars is None or original_chars <= max_chars:
                return kept

            rendered = _ensure_trailing_newline(kept) + MARK_AFTER
            return _append_notice(rendered, original_chars=original_chars, kept_content=kept, tail_mode=False)

        # Otherwise, read up to `line_count` lines.
        for _ in range(line_count):
            line = f.readline()
            if line == "":
                break
            original_chars += len(line)
            _keep_head(line)

        kept = "".join(kept_parts)
        if max_chars is None or original_chars <= max_chars:
            return kept

        rendered = _ensure_trailing_newline(kept) + MARK_AFTER
        return _append_notice(rendered, original_chars=original_chars, kept_content=kept, tail_mode=False)

