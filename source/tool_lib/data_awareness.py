import os
import fnmatch
from pathlib import Path
from typing import List, Optional
from langchain_core.tools import tool
from dotenv import load_dotenv





@tool
def scan_directory_structure(
    directory_path: str,
    max_depth: int = 3,
    max_files_per_dir: int = 5,
) -> str:
    """
    Scans a specified directory and returns a tree directory structure using standard ASCII characters.

    Parameters:
    -----------
    directory_path : str
        The absolute or relative path to the root directory to be scanned.
    max_depth : int, optional
        The maximum depth to traverse in the directory tree (default is 3).
    max_files_per_dir : int, optional
        The threshold for file count display. If a directory contains more files than this, 
        files are summarized (e.g., "[Summary: 150 .dcm files]") to save tokens (default is 5).

    Returns:
    --------
    str
        A string representing the directory tree using ASCII characters (|-- , +--).
        Example output:
        root/
        |-- sub-01/
        |   |-- func/
        |   |   +-- sub-01_task-rest_bold.nii.gz
        |   +-- anat/
        +-- README.md
    
    
    -----
    Note:
    To prevent excessive output (and LLM token explosion), this function enforces a global line limit
    internally: if the final rendered tree exceeds 200 lines, it keeps only the first 100 and last 100
    lines, and inserts a truncation notice indicating how many middle lines were removed.
    """

    # Global output limit for LLM safety (do NOT expose as a function parameter)
    MAX_TOTAL_LINES = 200
    HEAD_KEEP_LINES = 100
    TAIL_KEEP_LINES = 100

    root_path = Path(directory_path)

    if not root_path.exists():
        return f"Error: The path '{directory_path}' does not exist."
    if not root_path.is_dir():
        return f"Error: The path '{directory_path}' is not a directory."

    # Default ignore patterns
    ignore_patterns = [
        ".git",
        ".github",
        ".venv",
        "venv",
        "env",
        ".idea",
        ".vscode",
        "__pycache__",
        ".DS_Store",
        "*.pyc",
    ]

    tree_lines = []

    def _should_ignore(name: str) -> bool:
        """Check if a file/dir name matches any ignore pattern."""
        for pattern in ignore_patterns:
            if fnmatch.fnmatch(name, pattern):
                return True
        return False

    def _add_to_tree(path: Path, prefix: str = "", depth: int = 0):
        if depth > max_depth:
            return

        try:
            # Sort by type (dirs first) then name for consistent LLM reading
            items = sorted(list(path.iterdir()), key=lambda x: (not x.is_dir(), x.name))
        except PermissionError:
            tree_lines.append(f"{prefix}+-- [Permission Denied]")
            return

        # Filter ignored items
        items = [x for x in items if not _should_ignore(x.name)]

        # Split into dirs and files for processing logic
        dirs = [x for x in items if x.is_dir()]
        files = [x for x in items if x.is_file()]

        total_items = len(dirs) + len(files)
        processed_count = 0

        # 1. Process Directories
        for directory in dirs:
            processed_count += 1
            is_last_item = processed_count == total_items

            connector = "+-- " if is_last_item else "|-- "
            tree_lines.append(f"{prefix}{connector}{directory.name}/")

            # Prepare prefix for children
            child_prefix = prefix + ("    " if is_last_item else "|   ")
            _add_to_tree(directory, child_prefix, depth + 1)

        # 2. Process Files
        if len(files) > max_files_per_dir:
            extensions = {}
            for f in files:
                ext = f.suffix
                extensions[ext] = extensions.get(ext, 0) + 1

            summary_parts = [f"{count} {ext} files" for ext, count in extensions.items()]
            summary_text = ", ".join(summary_parts)

            connector = "+-- "
            tree_lines.append(f"{prefix}{connector}[Summary: {summary_text}]")
        else:
            for file in files:
                processed_count += 1
                is_last_item = processed_count == total_items

                connector = "+-- " if is_last_item else "|-- "
                tree_lines.append(f"{prefix}{connector}{file.name}")

    # Add root
    tree_lines.append(f"{root_path.name}/")
    _add_to_tree(root_path)

    # Post-trim to prevent token explosion:
    # If lines > 200, keep first 100 and last 100, insert a middle truncation notice.
    total_before = len(tree_lines)
    if total_before > MAX_TOTAL_LINES:
        removed = total_before - (HEAD_KEEP_LINES + TAIL_KEEP_LINES)
        
        notice = (
            f"... [Truncated: original output had {total_before} lines; to avoid excessive token usage, "
            f"{removed} middle lines were removed; only the first {HEAD_KEEP_LINES} lines and the last {TAIL_KEEP_LINES} lines are retained] ..."
        )

        English_version_notice = "Note: The original directory structure content is too long, so the middle part has been cut off to avoid token explosion. Only the head and tail of the directory structure are displayed."
        
        tree_lines = tree_lines[:HEAD_KEEP_LINES] + [notice] + tree_lines[-TAIL_KEEP_LINES:] + [English_version_notice]

    return "\n".join(tree_lines)






import nibabel as nib

@tool
def read_nifti_header(file_path: str) -> str:
    """
    Reads and returns the full raw header information of a NIfTI file. 
    
    Parameters:
    -----------
    file_path : str
        The absolute or relative path to the NIfTI file (.nii or .nii.gz).

    Returns:
    --------
    str
        The raw string representation of the NIfTI header containing all keys and values.
        Returns an error message string if the file cannot be found or read.
    """
    try:
        # Load the image header only (proxies the data)
        # nib.load is lightweight if we don't access the data array
        img = nib.load(file_path)
        
        # Return the standard string representation of the header object
        # This dumps all fields (sizeof_hdr, dim, pixdim, intent, etc.)
        return str(img.header)

    except Exception as e:
        return f"Error reading NIfTI header for '{file_path}': {str(e)}"



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
            result = subprocess.run(command, shell=True, capture_output=True, text=True, env=env, executable="/bin/bash")

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

        except Exception as e:
            # If there's a more fundamental error (e.g. Python environment, permissions)
            return f"Failed to run command: {command}\n\nError: {str(e)}"
    else:
        return "Invalid input. User aborted command execution."




import subprocess
from typing import Dict, Any

# @tool
def validate_bids(dataset_root: str) -> Dict[str, Any]:
    """
    Validate whether the dataset is compliant with the BIDS standard.
    
    Parameters:
    -----------
    - dataset_root (str): The absolute or relative path to the root directory of the BIDS dataset.

    Returns:
    --------
    - validation results.
    """

    try:

        # the test below is too strict, current sample dataset cannot pass it :(
        result = subprocess.run(
            ['bids-validator', dataset_root],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True
        )
        return {
            "bids_valid": result.returncode == 0,
            "bids_errors": result.stdout if result.returncode != 0 else None
        }

    except FileNotFoundError:
        return {
            "bids_valid": False,
            "bids_errors": "bids-validator not found. Please install via `npm install -g bids-validator`."
        }

