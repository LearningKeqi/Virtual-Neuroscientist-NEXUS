descriptions = [
    {
        "name": 'fmriprep_pipe',
        "module": 'tool_lib.fmriprep',
        "description": """
            Run fMRIPrep to preprocess specified subject. fMRIPrep can preprocess both sMRI and fMRI data.
        """,
        "detailed_schema": """
            **Parameters:**
            - bids_dir: Path to the raw, BIDS input dataset.
            - output_dir: Path to the output directory.
            - participant_label: Specific participant ID to process (without 'sub-').
            
            **Outputs:**
            ## For fMRI data:
            - Subject's BOLD files in MNI space: {output_dir}/sub-{participant_label}/ses-*/func/sub-{participant_label}_*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz, used for downstream analysis.
            - Mean BOLD file in MNI space: {output_dir}/sub-{participant_label}/ses-*/func/sub-{participant_label}_*_space-MNI152NLin2009cAsym_desc-preproc_bold_mean.nii.gz, used for quality control.
            - Subject's BOLD Brain Mask in MNI space: {output_dir}/sub-{participant_label}/ses-*/func/sub-{participant_label}_*_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz, used for quality control.

            ## For sMRI data:
            - Subject's bias-corrected t1w image in native space: {output_dir}/sub-{participant_label}/ses-1/anat/sub-{participant_label}_ses-1_desc-preproc_T1w.nii.gz.
            - Subject's t1w brain mask in native space: {output_dir}/sub-{participant_label}/ses-1/anat/sub-{participant_label}_ses-1_desc-brain_mask.nii.gz, used for quality control.
            - Subject's t1w hard tissue segmentation map in native space, including CSF, GM, and WM: {output_dir}/sub-{participant_label}/ses-1/anat/sub-{participant_label}_ses-1_dseg.nii.gz, used for quality control.
            - Subject's t1w normalized in MNI space: {output_dir}/sub-{participant_label}/ses-1/anat/sub-{participant_label}_ses-1_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz, used for quality control.
            - Subject's gray matter probability map normalized in MNI space: {output_dir}/sub-{participant_label}/ses-1/anat/sub-{participant_label}_ses-1_space-MNI152NLin2009cAsym_label-GM_probseg.nii.gz, this is only used for downstream analysis, not used for quality control.

        """,
    },
]
