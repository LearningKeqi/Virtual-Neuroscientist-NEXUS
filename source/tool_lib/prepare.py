from langchain_core.tools import tool
import subprocess
import os
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv


def get_subprocess_env():
    load_dotenv()

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

    return env


env = get_subprocess_env()


LOG_DIR = "Path/exp_result/temp_log"
os.makedirs(LOG_DIR, exist_ok=True)

TEMP_SBATCH_DIR = "Path/temp_sbatch_command"
os.makedirs(TEMP_SBATCH_DIR, exist_ok=True)


def truncate_head_tail(text: str, head_len: int = 100, tail_len: int = 200) -> str:
    if len(text) <= head_len + tail_len:
        return text
    return (
        text[:head_len]
        + "\n[...TRUNCATED...]\n"
        + text[-tail_len:]
    )



def run_command_orig(command: str) -> str:
    """
    Run a shell command, store full logs to a file, return a short summary string.

    Returns:
        A brief summary message suitable as an LLM observation, including:
          - Command that was executed
          - Truncated stdout/stderr or an indication if there's an error
          - Location of the saved log file
    """
    # Create a timestamped log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile_path = os.path.join(LOG_DIR, f"cmd_{timestamp}.log")

    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, env=env, executable="/bin/bash")

        # Write full stdout/stderr to logfile
        with open(logfile_path, "w") as f:
            f.write(f"Command executed: {command}\n\n")
            f.write(f"=== STDOUT ===\n{result.stdout}\n\n")
            f.write(f"=== STDERR ===\n{result.stderr}\n\n")
            f.write(f"Return code: {result.returncode}\n")

        # Summarize or truncate
        truncated_stdout = truncate_head_tail(result.stdout)
        truncated_stderr = truncate_head_tail(result.stderr)

        # Construct a short summary
        if result.returncode == 0:
            summary = (
                f"Command executed successfully:\n\n"
                f"{command}\n\n"
                f"STDOUT (truncated):\n{truncated_stdout}\n\n"
                f"STDERR (truncated):\n{truncated_stderr}\n\n"
                f"Full logs saved to: {logfile_path}"
            )
        else:
            summary = (
                f"Command executed but returned non-zero exit code: {result.returncode}\n\n"
                f"{command}\n\n"
                f"STDOUT (truncated):\n{truncated_stdout}\n\n"
                f"STDERR (truncated):\n{truncated_stderr}\n\n"
                f"Full logs saved to: {logfile_path}"
            )

        return summary

    except Exception as e:
        # If there's a more fundamental error (e.g. Python environment, permissions)
        return f"Failed to run command: {command}\n\nError: {str(e)}"


def run_command_sbatch(command: str) -> str:
    """
    Run a shell command via sbatch, wait for completion, and return its full output
    (stdout + stderr) as if it was run directly in the terminal.

    This function:
      - Creates a temporary sbatch script under TEMP_SBATCH_DIR
      - Submits it with `sbatch --wait`
      - Reads the Slurm output file and returns its contents
      - Cleans up the temporary script and log file
    """
    from uuid import uuid4
    uid = uuid4().hex

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_path = os.path.join(TEMP_SBATCH_DIR, f"cmd_{uid}.sh")
    temp_log_path = os.path.join(TEMP_SBATCH_DIR, f"cmd_{uid}.log")

    # Create the temporary sbatch script
    script_lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=temp_command",
        f"#SBATCH --output={temp_log_path}",
        "#SBATCH --gres=gpu:0",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=32GB",
        "",
        # The actual command to execute
        command,
    ]

    try:
        with open(script_path, "w") as f:
            f.write("\n".join(script_lines) + "\n")

        # # Make the script executable
        # os.chmod(script_path, 0o700)

        # Submit with sbatch and wait for completion
        sbatch_cmd = ["sbatch", "--wait", script_path]
        sbatch_result = subprocess.run(
            sbatch_cmd,
            capture_output=True,
            text=True,
            env=env,
        )

        # # If sbatch itself failed (e.g., no job submitted), surface its output
        # if sbatch_result.returncode != 0:
        #     return (
        #         "Failed to submit or run sbatch job.\n\n"
        #         f"Command: {' '.join(sbatch_cmd)}\n\n"
        #         f"STDOUT:\n{sbatch_result.stdout}\n\n"
        #         f"STDERR:\n{sbatch_result.stderr}"
        #     )

        # Read the log file produced by sbatch job – this is the "real" command output
        try:
            with open(temp_log_path, "r") as f:
                output = f.read()
        except FileNotFoundError:
            # If for some reason the log is missing, fall back to sbatch outputs
            return (
                f"Expected sbatch log file not found at: {temp_log_path}\n\n"
                f"sbatch STDOUT:\n{sbatch_result.stdout}\n\n"
                f"sbatch STDERR:\n{sbatch_result.stderr}"
            )

        return output

    finally:
        # Clean up temporary files
        for path in (script_path, temp_log_path):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                # Best-effort cleanup; ignore errors
                pass


def remove_nii_gz(filename):
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    else:
        return os.path.splitext(filename)[0]


def run_command(command: str) -> str:
    """
    Run a shell command directly (no secondary Slurm submission), wait for completion,
    and return its full output (stdout + stderr) as if it was run directly in the terminal.
    """
    
    result = subprocess.run(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        executable="/bin/bash",
    )

    return result.stdout or ""