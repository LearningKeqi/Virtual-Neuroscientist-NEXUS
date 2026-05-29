descriptions = [
    {
        "name": "smri_skull_stripping_qc",
        "module": "tool_lib.quality_control_lib",
        "description": "Perform quality control on the skull stripping step of structural MRI for a subject. You can choose to get the brain volume size in ml, or visually check the result.",
        "detailed_schema": """
            **Parameters:**
            - original_t1w_path: (str). Absolute path to the original, raw T1-weighted MRI image of a subject.
            - brain_mask_path: (str). Absolute path to the brain mask NIfTI file in the original, native t1w space.
            - metric_or_visual: (str). Choice of metric or visual QC. 'metric' or 'visual'.

            **Outputs:**
            - If metric_or_visual is 'metric', return the brain volume size in ml as a python dictionary with one key: {'brain_volume_size': <volume_in_ml>}
            - If metric_or_visual is 'visual', return the visual evaluation result of the skull stripping step.
        """
    },

    {
        "name": "smri_tissue_segmentation_qc",
        "module": "tool_lib.quality_control_lib",
        "description": """
            Perform quality control on the tissue segmentation step of structural MRI for a subject. 
            You can choose to get volume metrics of each tissue (CSF, GM, WM) type, or visually check the result.
        """,
        "detailed_schema": """
            **Parameters:**
            - original_t1w_path: (str). Absolute path to the original,raw T1-weighted NIfTI file.
            - brain_mask_path: (str). Absolute path to the brain mask NIfTI file in the original, native t1w space.
            - tissue_seg_path: (str). Absolute path to the hard brain tissue segmentation map in the original, native t1w space, where each voxel is assigned a discrete label corresponding to CSF, gray matter, or white matter.
            - metric_or_visual: (str). Choice of metric or visual QC. 'metric' or 'visual'.

            **Outputs:**
            - If metric_or_visual is 'metric', return the volume of each tissue type (CSF, GM, WM) in ml. Such as {'csf_ml': csf_volume, 'gm_ml': gm_volume, 'wm_ml': wm_volume}
            - If metric_or_visual is 'visual', return the visual evaluation result of the tissue segmentation step.
        """

    },

    {
        "name": "smri_to_mni_normalization_qc",
        "module": "tool_lib.quality_control_lib",
        "description": """
            Perform quality control on the t1w to mni normalization step of structural MRI. 
            You can choose to get Normalized Mutual Information (NMI) and Normalized Cross-correlation (NCC) between normalized t1w and mni, or visually check the result.
        """,
        "detailed_schema": """
            **Parameters:**
            - anat_in_mni: (str). Absolute path to the t1w image in mni space.
            - metric_or_visual: (str). Choice of metric or visual QC. 'metric' or 'visual'.

            **Outputs:**
            - If metric_or_visual is 'metric', return a dictionary containing Normalized Mutual Information (NMI) and Normalized Cross-correlation (NCC) between normalized t1w and mni, such as {'NMI_t1w2mni': nmi_value, 'NCC_t1w2mni': ncc_value}
            - If metric_or_visual is 'visual', return visual evaluation result of the t1w to mni normalization.
        """
    },
    {
        "name": "fmri_coregister_to_anat_qc",
        "module": "tool_lib.quality_control_lib",
        "description": """
            Perform quality control on the fmri to anat co-registration step.
            You can choose to get Normalized Mutual Information (NMI)  and Dice Similarity of brain mask between co-registered fmri and t1w, or visually check the result.
        """,
        "detailed_schema": """
            **Parameters:**
            - mean_fmri_registered_to_t1w: (str). Absolute path to the mean fMRI image coregistered to t1w.
            - mean_fmri_brain_mask_registered_to_t1w: (str). Absolute path to the brain mask of the mean fMRI image coregistered to t1w.
            - t1w_brain_image: (str). Absolute path to the original t1w image (with skull).
            - t1w_brain_mask: (str). Absolute path to the t1w brain mask NIfTI file in the original, native t1w space.
            - t1w_brain_seg: (str). Absolute path to the t1w brain segment, hard brain tissue segmentation map in original, native t1w space, where each voxel is assigned a discrete label corresponding to CSF, gray matter, or white matter.
            - metric_or_visual: (str). Choice of metric or visual QC. 'metric' or 'visual'.

            **Outputs:**
            - If metric_or_visual is 'metric', return a dictionary containing the Normalized Mutual Information (NMI)  and Dice Similarity of brain mask between co-registered fmri and t1w, {'NMI_fmri2t1w': nmi_value, 'Dice_fmri2t1w': dice_value}.
            - If metric_or_visual is 'visual', return the visual evaluation result of the fmri to anat co-registration.
        """
    },

    {
        "name": "fmri_normalized_to_mni_qc",
        "module": "tool_lib.quality_control_lib",
        "description": """
            Perform quality control on the fmri normalized to mni preprocessing step.
            You can choose to get the Normalized Mutual Information (NMI) and Dice Similarity of brain mask between normalized fmri and mni, or visually check the result.
        """,
        "detailed_schema": """
            **Parameters:**
            - mean_fmri_normalized_to_mni: (str). Absolute path to the mean fMRI image normalized to mni.
            - mean_fmri_brain_mask_normalized_to_mni: (str). Absolute path to the brain mask of the mean fMRI image normalized to mni.
            - metric_or_visual: (str). Choice of metric or visual QC. 'metric' or 'visual'.

            **Outputs:**
            - If metric_or_visual is 'metric', return the Normalized Mutual Information (NMI) and Dice Similarity of brain mask between normalized fmri and mni, {'NMI_fmri2mni': nmi_value, 'Dice_fmri2mni': dice_value}.
            - If metric_or_visual is 'visual', return the visual evaluation result of the fmri normalized to mni.  
        """
    }
]



