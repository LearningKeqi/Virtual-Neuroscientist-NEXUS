from .prepare import *

from typing import Optional
from langchain_core.tools import tool
import os



def fmriprep_pipe(
    bids_dir: str,
    output_dir: str,
    participant_label: Optional[str] = None,
) -> str:
    """
    Run fMRIPrep to preprocess specified subject. fMRIPrep can preprocess both sMRI and fMRI data.

    **Parameters:**
    - bids_dir: Path to the raw, BIDS input dataset.
    - output_dir: Path to the output directory.
    - participant_label: Specific participant ID to process (without 'sub-').
    
    **Outputs:**
    ## For fMRI data:
    - Subject's BOLD files in MNI space: {output_dir}/sub-{participant_label}/ses-*/func/sub-{participant_label}_*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
    - Mean BOLD file in MNI space: {output_dir}/sub-{participant_label}/ses-*/func/sub-{participant_label}_*_space-MNI152NLin2009cAsym_desc-preproc_bold_mean.nii.gz, used for quality control.
    - Subject's BOLD Brain Mask in MNI space: {output_dir}/sub-{participant_label}/ses-*/func/sub-{participant_label}_*_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz, used for quality control.

    ## For sMRI data:
    - Subject's bias-corrected t1w image in native space: {output_dir}/sub-{participant_label}/ses-1/anat/sub-{participant_label}_ses-1_desc-preproc_T1w.nii.gz.
    - Subject's t1w brain mask in native space: {output_dir}/sub-{participant_label}/ses-1/anat/sub-{participant_label}_ses-1_desc-brain_mask.nii.gz
    - Subject's t1w hard tissue segmentation map in native space, including CSF, GM, and WM: {output_dir}/sub-{participant_label}/ses-1/anat/sub-{participant_label}_ses-1_dseg.nii.gz
    - Subject's t1w normalized in MNI space: {output_dir}/sub-{participant_label}/ses-1/anat/sub-{participant_label}_ses-1_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz
    - Subject's gray matter probability map normalized in MNI space: {output_dir}/sub-{participant_label}/ses-1/anat/sub-{participant_label}_ses-1_space-MNI152NLin2009cAsym_label-GM_probseg.nii.gz
    """

    import os

    fmriprep_image = "Path/tools/fmriprep-24.1.1.sif"
    fs_license = "Path/tools/freesurfer/license.txt"

    if participant_label and participant_label.startswith("sub-"):
        participant_label = participant_label.replace("sub-", "")

    subject_id = participant_label if participant_label else "unknown"

    os.makedirs(output_dir, exist_ok=True)

    work_dir = os.path.join(output_dir, "work", f"sub-{subject_id}")
    os.makedirs(work_dir, exist_ok=True)

    fs_subjects_dir = os.path.join(output_dir, "freesurfer_subjects", f"sub-{subject_id}")
    os.makedirs(fs_subjects_dir, exist_ok=True)

    command = [
        "apptainer", "run", "--cleanenv",
        "-B", f"{bids_dir}:/data",
        "-B", f"{output_dir}:/out",
        "-B", f"{fs_license}:/fs_license",
        "-B", f"{work_dir}:/work",
        "-B", f"{fs_subjects_dir}:/fs_subjects",
        fmriprep_image, "/data", "/out", "participant",
    ]

    if participant_label:
        command += ["--participant-label", participant_label]

    command += [
        "--fs-license-file", "/fs_license",
        "--fs-subjects-dir", "/fs_subjects",
        "--output-spaces", "MNI152NLin2009cAsym",
        "--nthreads", "4",
        "--omp-nthreads", "4",
        "--stop-on-first-crash",
        "--mem", "32000",
        "--work-dir", "/work",
        "--skip-bids-validation",
    ]

    shell_command = " ".join(command)
    return run_command(shell_command)