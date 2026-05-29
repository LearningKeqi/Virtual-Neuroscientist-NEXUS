from .prepare import *
from langchain_core.tools import tool


# @tool
# need desc
def freesurfer_recon_all(t1w_file: str, subject_id: str, subjects_dir: str, autorecon_stage: Optional[str] = "all") -> str:
    """
    Perform structural MRI preprocessing using FreeSurfer's recon-all.

    **Parameters:**
    - t1w_file: Path to the T1-weighted input image.
    - subject_id: Unique identifier for the subject (used in FreeSurfer). Note that the subject ID should be a unique string identifier per subject, You need to adaptively extract the subject ID from the user's input and use it as a parameter for FreeSurfer.
    - subjects_dir: Path to the FreeSurfer SUBJECTS_DIR where output will be stored.
    - autorecon_stage: Stage of recon-all to run (e.g., "all", "autorecon1", "autorecon2", "autorecon3"). Default: "all".

    **Outputs:**
    - A processed FreeSurfer subject directory under the specified SUBJECTS_DIR.
    """


    os.makedirs(subjects_dir, exist_ok=True)
    env["SUBJECTS_DIR"] = subjects_dir

    command = (
        f"export SUBJECTS_DIR={subjects_dir} && "
        f"source $FREESURFER_HOME/SetUpFreeSurfer.sh && "
        f"recon-all -i {t1w_file} -s {subject_id} -{autorecon_stage}"
    )
    
    return run_command(command)
