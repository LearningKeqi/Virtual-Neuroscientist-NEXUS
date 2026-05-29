from .prepare import *
from langchain_core.tools import tool

import os
import shlex
from typing import Iterable, Set, List


def _afni_in_container(cmd: str, bind_dirs: Iterable[str]) -> str:
    """
    Run a shell command inside AFNI Apptainer image with necessary bind mounts.
    Assumes:
      - apptainer is available
      - env var AFNI_APPTAINER_IMAGE points to a valid .sif
    """
    q = shlex.quote
    image = os.environ["AFNI_APPTAINER_IMAGE"]

    binds: Set[str] = {os.path.abspath(d) for d in bind_dirs if d}
    bind_args = " ".join(f"--bind {q(d)}:{q(d)}" for d in sorted(binds))

    full = f"apptainer exec --cleanenv {bind_args} {q(image)} bash -lc {q(cmd)}"
    return run_command(full)



def remove_nii_gz(filename):
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    else:
        return os.path.splitext(filename)[0]





#---------- AFNI sMRI Tools--------------------
# @tool
# need desc
def afni_t1w_skull_strip(input_file: str, output_prefix: str) -> str:
    """
    Perform skull stripping on an sMRI image using AFNI's 3dSkullStrip.

    **Parameters:**
    - input_file: Path to the input anatomical t1w image.
    - output_prefix: Prefix for the output files, including the directory path.

    **Outputs:**
    - {output_prefix}_skull_stripped.nii.gz: Brain-extracted t1w file.
    - {output_prefix}_skull_stripped_mask.nii.gz: Brain mask of the brain-extracted t1w file.
    """

    q = shlex.quote

    input_file_abs = os.path.abspath(input_file)
    output_prefix_abs = os.path.abspath(output_prefix)

    output_file = f"{output_prefix_abs}_skull_stripped.nii.gz"
    brain_mask = f"{output_prefix_abs}_skull_stripped_mask_temp.nii.gz"
    brain_mask_bin = f"{output_prefix_abs}_skull_stripped_mask.nii.gz"

    output_dir = os.path.dirname(output_prefix_abs)
    os.makedirs(output_dir, exist_ok=True)

    input_dir = os.path.dirname(input_file_abs)
    output_parent = os.path.dirname(output_dir)

    cmds = [
        # f"mkdir -p {q(output_dir)}",
        f"3dSkullStrip -input {q(input_file_abs)} -prefix {q(brain_mask)} -mask_vol",
        f'3dcalc -a {q(brain_mask)} -expr "step(a)" -prefix {q(brain_mask_bin)}',
        f'3dcalc -a {q(input_file_abs)} -b {q(brain_mask_bin)} -expr "a*b" -prefix {q(output_file)}',
    ]

    logs = []
    for cmd in cmds:
        log = _afni_in_container(cmd, bind_dirs=[input_dir, output_parent, os.getcwd()])
        logs.append(log)

        # Check if the log indicates an error; if so, return the error message immediately.
        if ("returned non-zero exit code" in log) or ("Failed to run command" in log):
            return log


    return "\n\n".join(logs)




# @tool
# need desc
def afni_tissue_segmentation(input_file: str, output_dir: str) -> str:
    """
    Perform tissue segmentation on a skull-stripped sMRI image using AFNI's 3dSeg.

    Prerequisites: This tool must be used after t1w skull stripping, but the skull stripping step can be performed by any tool or software (like ANTs, FSL, etc.). It does not have to be AFNI 3dSkullStrip.

    **Parameters:**
    - input_file: Path to the input skull-stripped image (NIfTI format).
    - output_dir: Output directory path.

    **Outputs:**
    - {output_dir}/brain_seg.nii.gz: hard brain tissue segmentation map in which each voxel is assigned a discrete label corresponding to CSF, gray matter, or white matter, used for QC.
    - {output_dir}/gray_matter_prob.nii.gz: Gray matter probability map in native space.
    """


    input_file_abs = os.path.abspath(input_file)
    output_dir_abs = os.path.abspath(output_dir)


    output_gm_prob_file = f"{output_dir_abs}/gray_matter_prob.nii.gz"
    segsy_dir = f"{output_dir_abs}/Segsy"

    output_file = f"{output_dir_abs}/brain_seg.nii.gz"

    input_dir = os.path.dirname(input_file_abs) or os.getcwd()

    os.makedirs(output_dir_abs, exist_ok=True)

    cmds: List[str] = [
        # f"mkdir -p {shlex.quote(output_dir_abs)}",
        (
            "3dSeg "
            f"-anat {shlex.quote(input_file_abs)} "
            "-mask AUTO "
            "-classes 'CSF ; GM ; WM' "
            "-bias_classes 'GM ; WM' "
            "-bias_fwhm 25 "
            "-mixfrac UNI "
            "-main_N 5 "
            "-blur_meth BFT "
            f"-prefix {shlex.quote(output_dir_abs)}/Segsy -overwrite"
        ),
        (
            "3dAFNItoNIFTI "
            f"-prefix {shlex.quote(output_file)} "
            f"{shlex.quote(output_dir_abs)}/Segsy/Classes+orig"
        ),
        (
            "3dAFNItoNIFTI "
            f"-prefix {shlex.quote(output_gm_prob_file)} "
            f"'{segsy_dir}/Posterior+orig[1]'"
        ),
    ]

    logs = []
    for cmd in cmds:
        log = _afni_in_container(cmd, bind_dirs=[input_dir, output_dir_abs, os.getcwd()])
        logs.append(log)

        if ("returned non-zero exit code" in log) or ("Failed to run command" in log):
            return log

    return "\n\n".join(logs)


# need desc
def afni_normalize_t1w_to_mni(input_file: str, output_dir: str, subid: str) -> str:
    """
    Normalize sMRI image to a standard mni template space using AFNI's @SSwarper.

    **Parameters:**
    - input_file: Path to the raw t1w image (nii.gz format with skull).
    - output_dir: Directory where the output should be saved.
    - subid: subject id for the output files.

    **Outputs:**
    - {output_dir}/anatQQ.{subid}.nii: Normalized t1w image in mni template space.

    """


    input_file_abs = os.path.abspath(input_file)
    output_dir_abs = os.path.abspath(output_dir)
    input_dir = os.path.dirname(input_file_abs)
    os.makedirs(output_dir_abs, exist_ok=True)

    cmd = (
        "@SSwarper "
        f"-input  {shlex.quote(input_file_abs)} "
        "-base   Path/tools/mni_template/afni_mni_template/MNI152_2009_template_SSW.nii.gz "
        f"-subid  {shlex.quote(subid)} "
        f"-odir   {shlex.quote(output_dir_abs)} "
    )

    log = _afni_in_container(cmd, bind_dirs=[input_dir, output_dir_abs, os.getcwd()])
    return log



# need desc
def afni_warp_gm_tissue_seg_to_mni(path_to_gm_tissue_prob_map: str, t1w_in_mni_file: str, subid: str) -> str:
    """
    Warp the Gray Matter Probability Map of the tissue segmentation to MNI space using AFNI's 3dNwarpApply.

    Prerequisites: This tool must be used after 'afni_normalize_t1w_to_mni' tools.

    **Parameters:**
    - path_to_gm_tissue_prob_map: Path to the gray matter probability map file in native space.
    - t1w_in_mni_file: Path to the t1w image in MNI space generated by 'afni_normalize_t1w_to_mni' tool.
    - subid: subject id for the output files.

    **Outputs:**
    - {path_to_gm_tissue_prob_map.replace('.nii.gz', '_gm_mni.nii.gz')}: The Gray Matter Probability Map in MNI space, under the same directory as the original gray matter probability map file.

    """


    tissue_seg_base_path = os.path.dirname(path_to_gm_tissue_prob_map)

    t1w_in_mni_file_base_path = os.path.dirname(t1w_in_mni_file)
    warp_file = os.path.join(t1w_in_mni_file_base_path, f"anatQQ.{subid}_WARP.nii")
    affine_file = os.path.join(t1w_in_mni_file_base_path, f"anatQQ.{subid}.aff12.1D")
    
    master_file = t1w_in_mni_file
    # output_prefix = os.path.join(tissue_seg_base_path, "prob_GM_in_MNI.nii.gz")
    output_prefix = os.path.join(path_to_gm_tissue_prob_map.replace('.nii.gz', '_gm_mni.nii.gz'))

    cmd = (
        "3dNwarpApply "
        f"-source {shlex.quote(path_to_gm_tissue_prob_map)} "
        f"-nwarp {shlex.quote(warp_file)} {shlex.quote(affine_file)} "
        f"-master {shlex.quote(master_file)} "
        f"-interp wsinc5 "
        f"-prefix {shlex.quote(output_prefix)} "
    )




    log = _afni_in_container(cmd, bind_dirs=[tissue_seg_base_path, t1w_in_mni_file_base_path, os.getcwd()])
    return log



#---------- AFNI fMRI Tools--------------------

# @tool
# need desc
def afni_fmri_slice_timing_correction(input_file: str, output_file: str, tpattern: str = "alt+z") -> str:
    """
    Perform slice-timing correction on fMRI data using AFNI's 3dTshift.

    **Parameters:**
    - input_file: Path to the input fMRI file (.nii or nii.gz).
    - output_file: Path to save the slice-timing corrected file (.nii or nii.gz).
    - tpattern: Slice acquisition pattern (e.g., alt+z, seq+z). Default is alt+z.

    **Outputs:**
    - Time-shifted fMRI data saved at output_file.
    """

    # this slice timing function actually has many other hyperparameters, like slice ordering, tr etc.

    input_file_abs = os.path.abspath(input_file)
    output_file_abs = os.path.abspath(output_file)

    input_dir = os.path.dirname(input_file_abs)
    out_dir = os.path.dirname(output_file_abs)
    os.makedirs(out_dir, exist_ok=True)

    cmd = (
        # f"mkdir -p {shlex.quote(out_dir)} && "
        "3dTshift "
        f"-prefix {shlex.quote(output_file_abs)} "
        f"-tpattern {shlex.quote(tpattern)} "
        f"{shlex.quote(input_file_abs)}"
    )

    return _afni_in_container(cmd, bind_dirs=[input_dir, out_dir, os.getcwd()])



# need desc
def afni_fmri_extract_brain_mask(input_file: str, output_prefix: str):
    """
    Extract brain mask from fMRI data using AFNI's 3dAutomask.

    **Parameters:**
    - input_file: Path to the fMRI data.
    - output_prefix: Prefix for the output files, including the directory path.

    **Outputs:**
    - {output_prefix}_brain_mask.nii.gz: Brain mask generated from the fMRI data.
    """

    input_file_abs = os.path.abspath(input_file)
    output_prefix_abs = os.path.abspath(output_prefix)

    input_dir = os.path.dirname(input_file_abs)
    out_dir = os.path.dirname(output_prefix_abs)
    os.makedirs(out_dir, exist_ok=True)

    q = shlex.quote
    cmds = [
        # f"mkdir -p {q(out_dir)}",
        f"3dTstat -mean -prefix {q(output_prefix_abs)}_mean.nii.gz {q(input_file_abs)}",
        f"3dAutomask -prefix {q(output_prefix_abs)}_brain_mask.nii.gz {q(output_prefix_abs)}_mean.nii.gz",
    ]

    logs = []
    for cmd in cmds:
        log = _afni_in_container(cmd, bind_dirs=[input_dir, out_dir, os.getcwd()])
        logs.append(log)

        if ("returned non-zero exit code" in log) or ("Failed to run command" in log):
            return log


    return "\n\n".join(logs)




# @tool
# need desc
def afni_fmri_motion_correction(input_file: str, output_prefix: str, base_index: int = 0) -> str:
    """
    Perform motion correction using AFNI's 3dvolreg.

    **Parameters:**
    - input_file: Path to the input fMRI file (.nii or nii.gz).
    - output_prefix: Prefix for the output files, including the directory path.
    - base_index: Volume index to use as registration base (default: 0).

    **Outputs:**
    - {output_prefix}_motion_corrected.nii.gz: Motion-corrected fMRI data.
    - {output_prefix}_motion.1D: Motion parameters.
    - {output_prefix}_mc_fmri_brain_mask.nii.gz: Brain mask generated from the motion-corrected fMRI data, this is only used for Motion Correction Quality Control.
    """

    import numpy as np
    from pathlib import Path
    from typing import Union

    def align_afni_motion_to_fsl(
        motion_1d_path: Union[str, Path],
        motion_aligned_path: Union[str, Path],
    ) -> None:
        """

        Convert AFNI 3dvolreg motion.1D (roll pitch yaw dS dL dP) to an FSL-like
        motion parameter file with columns:
            [rot_x(rad), rot_y(rad), rot_z(rad), trans_x(mm), trans_y(mm), trans_z(mm)]

        Assumptions / conventions:
        - AFNI motion.1D columns are: roll pitch yaw dS dL dP
        where translations are in mm with +dS=Superior, +dL=Left, +dP=Posterior,
        and rotations are in degrees.
        - FSL-like output is interpreted in a common RAS world sense:
        +x=Right, +y=Anterior, +z=Superior, rotations in radians.
        - Therefore translations are mapped as:
            trans_x = -dL
            trans_y = -dP
            trans_z =  dS
        - Rotations are reordered to match axis naming:
            rot_x <- pitch
            rot_y <- yaw
            rot_z <- roll
        and converted deg -> rad.

        Notes:
        - If your NIfTI orientation is not RAS (axes flipped/permuted), this simple
        mapping may not match MCFLIRT's voxel-axis convention. In that case, you
        need a matrix-based conversion using image affines.

        Writes:
        - A whitespace-delimited text file at motion_aligned_path.
        """
        motion_1d_path = Path(motion_1d_path)
        motion_aligned_path = Path(motion_aligned_path)

        # Load: allow comments (#) and arbitrary whitespace
        data = np.loadtxt(motion_1d_path, comments="#")
        if data.ndim == 1:
            # Single-row edge case -> shape (1, 6)
            data = data.reshape(1, -1)

        if data.shape[1] != 6:
            raise ValueError(
                f"Expected 6 columns in AFNI motion.1D, got shape {data.shape} from {motion_1d_path}"
            )

        # AFNI columns
        roll_deg  = data[:, 0]
        pitch_deg = data[:, 1]
        yaw_deg   = data[:, 2]
        dS_mm     = data[:, 3]
        dL_mm     = data[:, 4]
        dP_mm     = data[:, 5]

        # Degrees -> radians
        deg2rad = np.pi / 180.0
        roll_rad  = roll_deg * deg2rad
        pitch_rad = pitch_deg * deg2rad
        yaw_rad   = yaw_deg * deg2rad

        # Reorder rotations to FSL-like [rot_x, rot_y, rot_z]
        rot_x = pitch_rad
        rot_y = yaw_rad
        rot_z = roll_rad

        # Map translations to RAS-like (+Right,+Anterior,+Superior)
        trans_x = -dL_mm
        trans_y = -dP_mm
        trans_z = dS_mm

        aligned = np.column_stack([rot_x, rot_y, rot_z, trans_x, trans_y, trans_z])

        # Ensure parent dir exists
        motion_aligned_path.parent.mkdir(parents=True, exist_ok=True)

        # Save with good precision
        np.savetxt(motion_aligned_path, aligned, fmt="%.8f", delimiter=" ")


    # ---- paths ----
    input_file_abs = os.path.abspath(input_file)
    output_prefix_abs = os.path.abspath(output_prefix)

    output_file = f"{output_prefix_abs}_motion_corrected.nii.gz"
    motion_file = f"{output_prefix_abs}_motion_original.1D"
    matrix_prefix = f"{output_prefix_abs}_mat.aff12.1D"

    out_dir = os.path.dirname(output_file)
    input_dir = os.path.dirname(input_file_abs)
    os.makedirs(out_dir, exist_ok=True)

    q = shlex.quote
    cmd = (
        # f"mkdir -p {q(out_dir)} && "
        "3dvolreg "
        f"-prefix {q(output_file)} "
        f"-base {int(base_index)} "
        f"-1Dfile {q(motion_file)} "
        f"-1Dmatrix_save {q(matrix_prefix)} "
        f"{q(input_file_abs)}"
    )

    # ---- run in container ----
    log = _afni_in_container(cmd, bind_dirs=[input_dir, out_dir, os.getcwd()])

    # ---- convert motion file on host side (after volreg produces it) ----
    motion_aligned_path = f"{output_prefix_abs}_motion.1D"
    align_afni_motion_to_fsl(motion_file, motion_aligned_path)

    brain_extract_input_file = f"{output_prefix_abs}_motion_corrected.nii.gz"
    brain_extract_output_prefix = f"{output_prefix_abs}_mc_fmri"
    afni_fmri_extract_brain_mask(input_file=brain_extract_input_file, output_prefix=brain_extract_output_prefix)

    return log





# need desc
def afni_coregister_fmri_to_anat(input_fmri_file: str, anat_file: str, output_dir: str) -> str:
    """
    Co-register fMRI to anatomical image using AFNI's (EPI to anat space).

    **Parameters:**
    - input_fmri_file: Path to the fMRI image with skull (.nii or nii.gz) to be co-registered to the anatomical image.
    - anat_file: Path to the original anatomical (T1w) image (.nii or nii.gz) .
    - output_dir: Directory where you want all outputs to be placed.

    **Outputs:**
    - {output_dir}/brain_seg.nii.gz: t1w brain segmentation result (CSF, GM, WM), used for quality control.
    - {output_dir}/t1w_brain_skull_stripped_mask.nii.gz: t1w brain mask, used for quality control.
    - {output_dir}/{input_fmri_file_name}_fmri_in_anat_mean.nii.gz: Mean fMRI image in anat space. input_fmri_file_name is the string after removing the .nii.gz suffix from input_fmri_file.
    - {output_dir}/{input_fmri_file_name}_fmri_in_anat_mean_brain_mask.nii.gz: Brain mask of the mean fMRI image in anat space, used for quality control.
    """

    input_fmri_file_abs = os.path.abspath(input_fmri_file)
    anat_file_abs = os.path.abspath(anat_file)
    output_dir_abs = os.path.abspath(output_dir)

    input_fmri_file_name = remove_nii_gz(os.path.basename(input_fmri_file_abs))
    # print(f"input_fmri_file_name: {input_fmri_file_name}")

    # QC: skullstrip + tissue seg (these functions are assumed already converted to apptainer)
    t1w_output_prefix = f"{output_dir_abs}/t1w_brain"
    brain_skull_stripped = f"{t1w_output_prefix}_skull_stripped.nii.gz"

    afni_t1w_skull_strip(input_file=anat_file_abs, output_prefix=t1w_output_prefix)
    afni_tissue_segmentation(input_file=brain_skull_stripped, output_dir=output_dir_abs)

    q = shlex.quote
    os.makedirs(output_dir_abs, exist_ok=True)

    cmd = (
        # f"mkdir -p {q(output_dir_abs)} && "
        f"cd {q(output_dir_abs)} && "
        "align_epi_anat.py "
        f"-anat {q(anat_file_abs)} "
        f"-epi {q(input_fmri_file_abs)} "
        "-epi_base mean "
        "-epi2anat "
        "-cost lpc+ZZ "
        "-giant_move "
        "-partial_coverage "
        "-check_flip "
        "-save_skullstrip "
        "-keep_rm_files "
        "-overwrite && "
        f"3dAFNItoNIFTI -prefix {q(input_fmri_file_name)}_fmri_in_anat.nii.gz {q(input_fmri_file_name)}_al+orig && "
        f"3dTstat -mean -prefix {q(input_fmri_file_name)}_fmri_in_anat_mean_temp.nii.gz {q(input_fmri_file_name)}_fmri_in_anat.nii.gz && "
        f"3dresample -master {q(anat_file_abs)} -input {q(input_fmri_file_name)}_fmri_in_anat_mean_temp.nii.gz "
        f"-prefix {q(input_fmri_file_name)}_fmri_in_anat_mean.nii.gz && "
        f"3dAutomask -prefix {q(input_fmri_file_name)}_fmri_in_anat_mean_brain_mask.nii.gz "
        f"{q(input_fmri_file_name)}_fmri_in_anat_mean.nii.gz"
    )

    anat_dir = os.path.dirname(anat_file_abs)
    fmri_dir = os.path.dirname(input_fmri_file_abs)

    return _afni_in_container(cmd, bind_dirs=[anat_dir, fmri_dir, output_dir_abs, os.getcwd()])


# need desc
def afni_normalize_fmri_to_mni(fmri_file: str, t1w_file: str, afni_normalized_fmri_output_dir: str, afni_normalized_t1w_output_dir: str, subid: str):
    """
    Normalize fMRI to MNI space using AFNI's afni_proc.py. 

    Use this tool, you do not need 'afni_coregister_fmri_to_anat' as prerequisite. This tool will internally perform the coregistration to the T1w image.

    Prerequisites: This tool must be used after the `afni_normalize_t1w_to_mni` tool, which was used to normalize the T1w image to MNI space with AFNI.

    **Parameters:**
    - fmri_file: Path to the fMRI image with skull.
    - t1w_file: Path to the T1w image with skull.
    - afni_normalized_fmri_output_dir: Directory where the normalized fMRI image obtained from AFNI will be saved. This directory must not be already exist, otherwise will cause an error. This function will automatically create this directory.
    - afni_normalized_t1w_output_dir: Directory where the normalized T1w image was saved.
    - subid: Subject ID. Must be the same as the subid used in the `afni_normalize_t1w_to_mni` function.

    **Outputs:**
    - {afni_normalized_fmri_output_dir}/{subid}_normalized_fmri_in_mni.nii.gz: Normalized fMRI image in MNI space.
    - {afni_normalized_fmri_output_dir}/{subid}_normalized_fmri_in_mni_mean.nii.gz: Mean normalized fMRI image in MNI space.
    - {afni_normalized_fmri_output_dir}/{subid}_normalized_fmri_in_mni_brain_mask.nii.gz: Brain mask of the normalized fMRI image in MNI space, used for quality control.
    - {afni_normalized_fmri_output_dir}/{subid}_fmri_mean_in_t1wspace.nii.gz: Mean fMRI image in t1w space, used for quality control.
    - {afni_normalized_fmri_output_dir}/{subid}_fmri_brain_mask_in_t1wspace.nii.gz: Brain mask of the fMRI image in t1w space, used for quality control.
    - {afni_normalized_fmri_output_dir}/t1w_brain_skull_stripped_mask.nii.gz: t1w brain mask in the original t1w space, used for quality control.
    - {afni_normalized_fmri_output_dir}/brain_seg.nii.gz: t1w brain segmentation result (CSF, GM, WM) in the original t1w space, used for quality control.
    """

    if os.path.exists(afni_normalized_fmri_output_dir):
        return (
            f"Directory {afni_normalized_fmri_output_dir} already exists, please delete it or rename "
            f"the existing directory first safely. This tool requires a new directory path."
        )

    q = shlex.quote

    fmri_file_abs = os.path.abspath(fmri_file)
    t1w_file_abs = os.path.abspath(t1w_file)
    fmri_dir = os.path.dirname(fmri_file_abs)
    t1w_dir = os.path.dirname(t1w_file_abs)

    out_dir_abs = os.path.abspath(afni_normalized_fmri_output_dir)
    out_parent = os.path.dirname(out_dir_abs)

    t1w_norm_dir_abs = os.path.abspath(afni_normalized_t1w_output_dir)
    t1w_norm_parent = os.path.dirname(t1w_norm_dir_abs)

    cmd_proc = (
        "afni_proc.py "
        f"-subj_id  {q(subid)} "
        f"-dsets    {q(fmri_file_abs)} "
        f"-copy_anat  {q(t1w_norm_dir_abs)}/anatSS.{q(subid)}.nii "
        "-anat_has_skull  no "
        f"-anat_follower   anat_w_skull anat {q(t1w_norm_dir_abs)}/anatU.{q(subid)}.nii "
        "-blocks   tshift align tlrc volreg blur mask scale regress "
        "-align_unifize_epi  local "
        "-align_opts_aea     -cost lpc+ZZ -giant_move -check_flip "
        "-tlrc_base          Path/tools/mni_template/afni_mni_template/MNI152_2009_template_SSW.nii.gz "
        "-tlrc_NL_warp "
        "-tlrc_NL_warped_dsets "
        f"{q(t1w_norm_dir_abs)}/anatQQ.{q(subid)}.nii "
        f"{q(t1w_norm_dir_abs)}/anatQQ.{q(subid)}.aff12.1D "
        f"{q(t1w_norm_dir_abs)}/anatQQ.{q(subid)}_WARP.nii "
        "-volreg_align_to    MIN_OUTLIER "
        "-volreg_align_e2a "
        "-volreg_tlrc_warp "
        "-volreg_warp_dxyz   2 "
        f"-out_dir {q(out_dir_abs)} "
        "-html_review_style none "
        "-execute"
    )

    logs = []
    logs.append(
        _afni_in_container(
            cmd_proc,
            bind_dirs=[fmri_dir, t1w_dir, t1w_norm_parent, out_parent, os.getcwd()]
        )
    )

    t1w_output_prefix = f"{out_dir_abs}/t1w_brain"
    afni_t1w_skull_strip(input_file=t1w_file_abs, output_prefix=t1w_output_prefix)

    brain_skull_stripped = f"{t1w_output_prefix}_skull_stripped.nii.gz"
    afni_tissue_segmentation(input_file=brain_skull_stripped, output_dir=out_dir_abs)

    cmds_post = [
        # MNI space 4D fMRI nifti
        (
            f"3dAFNItoNIFTI -prefix {q(out_dir_abs)}/{q(subid)}_normalized_fmri_in_mni.nii.gz "
            f"{q(out_dir_abs)}/pb02.{q(subid)}.r01.volreg+tlrc"
        ),

        # mean in MNI
        (
            f"3dTstat -mean -prefix {q(out_dir_abs)}/{q(subid)}_normalized_fmri_in_mni_mean.nii.gz "
            f"{q(out_dir_abs)}/{q(subid)}_normalized_fmri_in_mni.nii.gz"
        ),

        # mask of mean in MNI
        (
            f"3dAutomask -prefix {q(out_dir_abs)}/{q(subid)}_normalized_fmri_in_mni_brain_mask.nii.gz "
            f"{q(out_dir_abs)}/{q(subid)}_normalized_fmri_in_mni_mean.nii.gz"
        ),

        # orig fmri mean (native space)
        (
            f"3dTstat -mean -prefix {q(out_dir_abs)}/{q(subid)}_orig_fmri_mean.nii.gz "
            f"{q(fmri_file_abs)}"
        ),

        # invert affine (EPI->anat) and save
        (
            f"cat_matvec {q(out_dir_abs)}/anatSS.{q(subid)}_al_junk_mat.aff12.1D -I -ONELINE "
            f"> {q(out_dir_abs)}/fmri_to_t1w.aff12.1D"
        ),

        # apply affine to orig mean -> mean in t1w space
        (
            "3dAllineate "
            f"-input  {q(out_dir_abs)}/{q(subid)}_orig_fmri_mean.nii.gz "
            f"-base   {q(t1w_file_abs)} "
            f"-master {q(t1w_file_abs)} "
            f"-1Dmatrix_apply {q(out_dir_abs)}/fmri_to_t1w.aff12.1D "
            "-final  wsinc5 "
            f"-prefix {q(out_dir_abs)}/{q(subid)}_fmri_mean_in_t1wspace.nii.gz "
            "-overwrite"
        ),

        # mask in t1w space
        (
            f"3dAutomask -prefix {q(out_dir_abs)}/{q(subid)}_fmri_brain_mask_in_t1wspace.nii.gz "
            f"{q(out_dir_abs)}/{q(subid)}_fmri_mean_in_t1wspace.nii.gz"
        ),
    ]

    for c in cmds_post:
        logs.append(
            _afni_in_container(
                c,
                bind_dirs=[fmri_dir, t1w_dir, t1w_norm_parent, out_parent, os.getcwd()]
            )
        )


    os.remove(f"Path/proc.{subid}")
    os.remove(f"Path/output.proc.{subid}")



    return "\n\n".join(logs)




# @tool
# need desc
def afni_spatial_smoothing(input_file: str, output_file: str, fwhm: float = 2.0) -> str:
    """
    Apply spatial smoothing using AFNI's 3dmerge.

    **Parameters:**
    - input_file: Path to input fmri.
    - output_file: Path to save smoothed fMRI file (.nii or nii.gz).
    - fwhm: Full-width half maximum for Gaussian kernel (default: 2.0 mm).

    **Outputs:**
    - Spatially smoothed fMRI file saved at output_file (.nii or nii.gz).
    """

    q = shlex.quote

    input_abs = os.path.abspath(input_file)
    output_abs = os.path.abspath(output_file)

    in_dir = os.path.dirname(input_abs)
    out_dir = os.path.dirname(output_abs)
    out_parent = os.path.dirname(out_dir)

    os.makedirs(out_dir, exist_ok=True)

    cmd = (
        # f"mkdir -p {q(out_dir)} && "
        f"3dmerge -1blur_fwhm {float(fwhm)} -doall -prefix {q(output_abs)} {q(input_abs)}"
    )

    return _afni_in_container(cmd, bind_dirs=[in_dir, out_parent, os.getcwd()])



# @tool
# need desc
def afni_temporal_filter_fmri(input_file: str, output_file: str, motion_file: str, low_freq: float = 0.01, high_freq: float = 0.1, polort: int = 2) -> str:
    """
    Preprocess fMRI signal by regressing out motion, trends, and applying bandpass filtering using AFNI's 3dTproject.

    Prerequisites: This tool must be used after the `afni_fmri_motion_correction` tool.

    **Parameters:**
    - input_file: Path to fMRI data.
    - output_file: Path to save the temporally filtered fMRI file (.nii or nii.gz).
    - motion_file: 1D file containing motion regressors (e.g., from afni_fmri_motion_correction).
    - low_freq, high_freq: Bandpass range in Hz.
    - polort: Polynomial detrending order.

    **Outputs:**
    - Denoised, bandpass-filtered fMRI data specified by output_file (.nii or nii.gz).
    """
    q = shlex.quote

    input_abs = os.path.abspath(input_file)
    output_abs = os.path.abspath(output_file)
    motion_abs = os.path.abspath(motion_file)
    
    if 'original' not in motion_abs:
        motion_abs = motion_abs.replace('.1D', '_original.1D')

    in_dir = os.path.dirname(input_abs)
    motion_dir = os.path.dirname(motion_abs)

    out_dir = os.path.dirname(output_abs)
    out_parent = os.path.dirname(out_dir)

    os.makedirs(out_dir, exist_ok=True)

    cmd = (
        # f"mkdir -p {q(out_dir)} && "
        "3dTproject "
        f"-input {q(input_abs)} "
        f"-ort {q(motion_abs)} "
        f"-polort {int(polort)} "
        f"-bandpass {float(low_freq)} {float(high_freq)} "
        f"-prefix {q(output_abs)}"
    )

    return _afni_in_container(cmd, bind_dirs=[in_dir, motion_dir, out_parent, os.getcwd()])






def _afni_in_container_no_sbatch(cmd: str, bind_dirs: Iterable[str]) -> str:
    """
    Run a shell command inside AFNI Apptainer image with necessary bind mounts.
    Assumes:
      - apptainer is available
      - env var AFNI_APPTAINER_IMAGE points to a valid .sif
    """
    q = shlex.quote
    image = os.environ["AFNI_APPTAINER_IMAGE"]

    binds: Set[str] = {os.path.abspath(d) for d in bind_dirs if d}
    bind_args = " ".join(f"--bind {q(d)}:{q(d)}" for d in sorted(binds))

    full = f"apptainer exec --cleanenv {bind_args} {q(image)} bash -lc {q(cmd)}"
    return run_command_no_sbatch(full)




def afni_normalize_t1w_to_mni_no_sbatch(input_file: str, output_dir: str, subid: str) -> str:
    """
    Normalize sMRI image to a standard mni template space using AFNI's @SSwarper.

    **Parameters:**
    - input_file: Path to the raw t1w image (nii.gz format with skull).
    - output_dir: Directory where the output should be saved.
    - subid: subject id for the output files.

    **Outputs:**
    - {output_dir}/anatQQ.{subid}.nii: Normalized t1w image in mni template space.

    """


    input_file_abs = os.path.abspath(input_file)
    output_dir_abs = os.path.abspath(output_dir)
    input_dir = os.path.dirname(input_file_abs)
    os.makedirs(output_dir_abs, exist_ok=True)

    cmd = (
        "@SSwarper "
        f"-input  {shlex.quote(input_file_abs)} "
        "-base   Path/tools/mni_template/afni_mni_template/MNI152_2009_template_SSW.nii.gz "
        f"-subid  {shlex.quote(subid)} "
        f"-odir   {shlex.quote(output_dir_abs)} "
    )

    log = _afni_in_container_no_sbatch(cmd, bind_dirs=[input_dir, output_dir_abs, os.getcwd()])
    return log
