from .prepare import *

# ---------- FSL TOOLS ---------- #
# ---------- FSL TOOLS - sMRI ---------- #

#@tool
# need desc
def fsl_bet_t1w(input_file: str, output_prefix: str, frac: Optional[float] = 0.3) -> str:
    """
    Perform brain extraction (skull stripping) on an sMRI image using FSL BET.

    **Parameters:**
    - input_file: Path to the T1w image with skull.
    - output_prefix: Full path prefix (including directory) for the files (skull-stripped t1w and brain mask).
    - frac: Fractional intensity threshold (default: 0.3). Higher values means more aggressive skull stripping (tend to remove more parts).

    **Outputs:**
    - {output_prefix}_brain.nii.gz: Skull-stripped T1w image.
    - {output_prefix}_brain_mask.nii.gz: Brain mask of the skull-stripped T1w image.
    """

    output_brain = f"{output_prefix}_brain.nii.gz"
    out_dir = os.path.dirname(output_brain)

    command = f"mkdir -p {out_dir} && bet {input_file} {output_brain} -f {frac} -g 0 -m"
    return run_command(command)



#@tool
# need desc
def fsl_fast(skull_stripped_t1w_file: str, output_prefix: str) -> str:
    """
    Perform tissue segmentation on a skull-stripped structural image using FSL FAST.

    Prerequisites: This tool must be used after t1w skull stripping, but the skull stripping step can be performed by any tool or software (like AFNI, ANTs, etc.). It does not have to be FSL BET.

    **Parameters:**
    - skull_stripped_t1w_file: Path to a skull-stripped T1-weighted NIfTI image.
    - output_prefix: Full path prefix (including directory) for all output files.

    **Outputs:**
    - CSF, GM, and WM partial volume maps and segmentation labels with the specified prefix. 
    - {output_prefix}_pve_0/1/2.nii.gz, specifically {output_prefix}_pve_1.nii.gz is the gray matter probability map in native space.
    - {output_prefix}_seg.nii.gz: hard brain tissue segmentation map in which each voxel is assigned a discrete label corresponding to CSF, gray matter, or white matter, used for QC.
    """
    out_dir = os.path.dirname(output_prefix)

    command = f"mkdir -p {out_dir} && fast -o {output_prefix} {skull_stripped_t1w_file}"
    return run_command(command)




#@tool
# need desc
def fsl_normalize_t1w_to_mni(
    skull_stripped_t1w_file: str,
    full_head_t1w_file: str,
    output_prefix: str,
) -> str:
    """
    Normalize T1w image to MNI space using FSL.

    Prerequisites: This tool must be used after t1w skull stripping, but the skull stripping step can be performed by any tool or software (like AFNI, ANTs, etc.).

    **Parameters:**
    - skull_stripped_t1w_file: Path to skull-stripped (brain-only) T1-weighted image.
    - full_head_t1w_file: Path to full-head (with skull) T1-weighted image.
    - output_prefix: Full path prefix (including directory) for all output files.

    **Outputs:**
    - {prefix}_anat2mni_affine.mat: Affine matrix from (brain) T1w to MNI brain (FLIRT).
    - {prefix}_anat2mni_warpcoef.nii.gz: Nonlinear warp coefficients (FNIRT --cout).
    - {prefix}_anat_mni_brain.nii.gz: Skull-stripped T1w warped into MNI brain space.
    """

    # Outputs
    affine_mat = f"{output_prefix}_anat2mni_affine.mat"
    warpcoef = f"{output_prefix}_anat2mni_warpcoef.nii.gz"
    anat_mni_brain = f"{output_prefix}_anat_mni_brain.nii.gz"

    mni_template_resolution = '1mm'


    if '2' in mni_template_resolution:
        mni_brain = 'Path/tools/mni_template/mni_template_2mm/brain.nii.gz'
        mni_whole = 'Path/tools/mni_template/mni_template_2mm/with_skull.nii.gz'
        mni_brain_mask = 'Path/tools/mni_template/mni_template_2mm/brain_mask.nii.gz'
    elif '1' in mni_template_resolution:
        mni_brain = 'Path/tools/mni_template/mni_template_1mm_aligned/mni152_t1w_brain.nii.gz'
        mni_whole = 'Path/tools/mni_template/mni_template_1mm_aligned/mni152_t1w_withskull.nii.gz'
        mni_brain_mask = 'Path/tools/mni_template/mni_template_1mm_aligned/mni152_brain_mask.nii.gz'

    # FNIRT config
    fnirt_config = "T1_2_MNI152_2mm"

    out_dir = os.path.dirname(output_prefix)

    cmds = [
        f"mkdir -p {out_dir}", 
        # 1) FLIRT: brain -> MNI brain
        (
            f"flirt -in {skull_stripped_t1w_file} "
            f"-ref {mni_brain} "
            f"-omat {affine_mat} "
            f"-dof 12"
        ),

        # 2) FNIRT: full-head -> MNI whole-head (estimate warp)
        # Try brain_mask first; if it doesn't exist on the system, you can swap to brain_mask_dil.
        (
            f"fnirt --in={full_head_t1w_file} "
            f"--aff={affine_mat} "
            f"--ref={mni_whole} "
            f"--config={fnirt_config} "
            f"--refmask={mni_brain_mask} "
            f"--cout={warpcoef}"
        ),

        # 3) applywarp: brain -> MNI brain (final output)
        (
            f"applywarp --in={skull_stripped_t1w_file} "
            f"--ref={mni_brain} "
            f"--warp={warpcoef} "
            f"--out={anat_mni_brain}"
        )
    ]

    logs = []
    for cmd in cmds:
        log = run_command(cmd)
        logs.append(log)
        # Check if the log indicates an error; if so, return immediately.
        if ("returned non-zero exit code" in log) or ("Failed to run command" in log):
            return log  # Stop processing further commands and return the error message.
        
    return "\n\n".join(logs)





def fsl_warp_gm_tissue_seg_to_mni(
    path_to_gm_tissue_prob_map: str,
    warpcoef: str
    ) -> str:
    """
    Warp the Gray Matter Probability Map of the tissue segmentation to MNI space using FSL.

    Prerequisites: This tool must be used after 'fsl_normalize_t1w_to_mni' tool.

    **Parameters:**
    - path_to_gm_tissue_prob_map: Path to the gray matter probability map file in native space.
    - warpcoef: Nonlinear warp coefficients file generated by 'fsl_normalize_t1w_to_mni' tool.

    **Outputs:**
    - {path_to_gm_tissue_prob_map.replace('.nii.gz', '_gm_mni.nii.gz')}: The Gray Matter Probability Map in MNI space, under the same directory as the original gray matter probability map file.
    
    """
    gm_native = path_to_gm_tissue_prob_map
    gm_mni = path_to_gm_tissue_prob_map.replace('.nii.gz', '_gm_mni.nii.gz')
    out_dir = os.path.dirname(gm_mni)

    mni_brain = 'Path/tools/mni_template/mni_template_1mm_aligned/mni152_t1w_brain.nii.gz'

    cmds = [
        f"mkdir -p {out_dir}",
        (
            f"applywarp "
            f"--in={gm_native} "
            f"--ref={mni_brain} "
            f"--warp={warpcoef} "
            f"--out={gm_mni} "
            f"--interp=trilinear"
        )
    ]

    logs = []
    for cmd in cmds:
        log = run_command(cmd)
        logs.append(log)
        if ("returned non-zero exit code" in log) or ("Failed to run command" in log):
            return log

    return "\n\n".join(logs)








# ---------- FSL TOOLS - fMRI ---------- #
#@tool
# need desc
def fsl_skullstrip_fmri(input_file: str, output_prefix: str, frac: float = 0.3) -> str:
    """
    Perform brain extraction on a 4D fMRI image using FSL BET.

    **Parameters:**
    - input_file: Path to the 4D fMRI NIfTI image.
    - output_prefix: Full path prefix (including directory) for the files (skull-stripped fmri and brain mask).
    - frac: Fractional intensity threshold for BET (default: 0.3).

    **Outputs:**
    - {output_prefix}_skull_stripped_fmri.nii.gz: Skull-stripped fMRI image.
    - {output_prefix}_brain_mask.nii.gz: Brain mask of the skull-stripped fMRI image in the original fmri space.
    """

    mean_img = f"{output_prefix}_mean.nii.gz"
    bet_output = f"{output_prefix}_mean_skull_stripped.nii.gz"
    bet_mask = f"{output_prefix}_mean_skull_stripped_mask.nii.gz"

    skull_stripped_fmri = f"{output_prefix}_skull_stripped_fmri.nii.gz"
    brain_mask_path = f"{output_prefix}_brain_mask.nii.gz"

    out_dir = os.path.dirname(output_prefix)

    cmds = [
        f"mkdir -p {out_dir}",
        f"fslmaths {input_file} -Tmean {mean_img}",
        f"bet {mean_img} {bet_output} -f {frac} -m",
        f"fslmaths {input_file} -mas {bet_mask} {skull_stripped_fmri}",
        f"cp -f {bet_mask} {brain_mask_path}"
    ]

    logs = []
    for cmd in cmds:
        log = run_command(cmd)
        logs.append(log)
        # Check if the log indicates an error; if so, return the error message immediately.
        if ("returned non-zero exit code" in log) or ("Failed to run command" in log):
            return log
    return "\n\n---\n\n".join(logs)




#@tool
# need desc
def fsl_slicetimer(input_file: str, output_file: str, TR: float = 2.0) -> str:
    """
    Slice-timing correction for fMRI using FSL Slicetimer.

    **Parameters:**
        input_file: Path to the 4D fMRI input file.
        output_file: Path to save the slice-time corrected file.
        TR: TR (time of repetition) in seconds. Default is 2.0.
    
    **Outputs:**
    - A 4D slice-time-corrected fMRI image.
    """
    out_dir = os.path.dirname(output_file)
    command = f"mkdir -p {out_dir} && slicetimer -i {input_file} -o {output_file} --repeat={TR}"
    return run_command(command)



#@tool
# need desc
def fsl_motion_correct(input_file: str, output_file: str) -> str:
    """
    Perform motion correction on a 4D fMRI image using FSL MCFLIRT.

    **Parameters:**
    - input_file: Path to the input fMRI NIfTI image.
    - output_file: Path to save the motion-corrected image.

    **Outputs:**
    - output_file: Path to the motion-corrected fMRI image.
    - {output_file}.par: Motion parameters.
    - A brain mask of the motion-corrected fMRI in the same directory as the output_file. named as 'mc_fmri_brain_mask.nii.gz'. This is only used for Motion Correction Quality Control.

    """
    out_dir = os.path.dirname(output_file)
    command = f"mkdir -p {out_dir} && mcflirt -in {input_file} -out {output_file} -plots -report"

    log = run_command(command)

    input_file_mc = output_file
    output_prefix_mc = os.path.dirname(output_file) + '/mc_fmri'
    # print(f"output_prefix_mc: {output_prefix_mc}")
    # print(f"input_file_mc: {input_file_mc}")
    fsl_skullstrip_fmri(input_file_mc, output_prefix_mc)

    return log






# need desc
def fsl_coregister_func_to_anat(
    fmri_file: str,
    t1w_file: str,
    output_prefix: str,
) -> str:
    """
    Co-register fMRI (EPI) to T1w structural image using FSL.

    **Parameters:**
    - fmri_file: Path to the 4D fMRI image with skull.
    - t1w_file: Path to the T1-weighted structural image with skull (whole head).
    - output_prefix: Full path prefix (including directory) for output files, e.g. "{output_prefix}_func2anat.mat"

    **Outputs:**
    - {output_prefix}_func2anat.mat: Affine matrix from mean fMRI (EPI) to T1w image.
    - {output_prefix}_mean_func2anat.nii.gz: Mean fMRI in T1w space.
    - {output_prefix}_mean_func2anat_brain_mask.nii.gz: Brain mask of the mean fMRI in T1w space, used for quality control.
    - {output_prefix}_t1w_proc.anat/T1_biascorr_brain_mask.nii.gz: Brain mask of the T1w image used for quality control.
    - {output_prefix}_t1w_proc.anat/T1_fast_seg.nii.gz: Segmentation of the T1w image used for quality control.
    """

    # Intermediate / output paths
    fmri_mc = f"{output_prefix}_fmri_mc.nii.gz"
    mean_func = f"{output_prefix}_mean_original_space.nii.gz"

    # fsl_anat output directory
    anat_out = f"{output_prefix}_t1w_proc"
    anat_dir = f"{anat_out}.anat"
    t1_biascorr = f"{anat_dir}/T1_biascorr.nii.gz"
    t1_brain = f"{anat_dir}/T1_biascorr_brain.nii.gz"
    # t1_brain_mask = f"{anat_dir}/T1_biascorr_brain_mask.nii.gz"

    # epi_reg output prefix
    epi_reg_out = f"{output_prefix}_mean_func2anat"
    # aligned_mean_func = f"{epi_reg_out}.nii.gz"
    mat_file = f"{epi_reg_out}.mat"

    # Final requested deliverables
    # final_mean_brain = f"{output_prefix}_mean_func2anat_brain.nii.gz"
    final_mat = f"{output_prefix}_func2anat.mat"

    out_dir = os.path.dirname(output_prefix)

    cmds = [
        f"mkdir -p {out_dir}",
        # 1) Motion correction (more stable mean reference)
        f"mcflirt -in {fmri_file} -out {fmri_mc}",

        # 2) Mean EPI reference
        f"fslmaths {fmri_mc} -Tmean {mean_func}",

        # 3) T1 processing (bias-correct + brain + mask + segmentation etc.)
        f"fsl_anat -i {t1w_file} -o {anat_out} --clobber --noreorient --nocrop ",

        # 4) EPI(mean) -> T1 registration using epi_reg (BBR)
        f"epi_reg --epi={mean_func} --t1={t1_biascorr} --t1brain={t1_brain} --out={epi_reg_out}",

        # # 5) Brain-mask the registered mean EPI in T1 space
        # f"fslmaths {aligned_mean_func} -mas {t1_brain_mask} {final_mean_brain}",

        # 6) Copy matrix to the requested name
        f"cp -f {mat_file} {final_mat}",

        f"bet {output_prefix}_mean_func2anat.nii.gz {output_prefix}_mean_func2anat_brain.nii.gz -f 0.4 -m",
    ]


    logs = []
    for cmd in cmds:
        log = run_command(cmd)
        logs.append(log)
        # Check if the log indicates an error; if so, return the error message immediately.
        if ("returned non-zero exit code" in log) or ("Failed to run command" in log):
            return log

    return "\n\n".join(logs)




# need desc
def fsl_normalize_func_to_mni(
    fmri_file: str,
    t1w_file: str,      
    func2struct_mat: str,
    output_prefix: str
):
    """
    Normalize fMRI to MNI space using FSL.

    Prerequisites: This function **must** be used after the `fsl_coregister_func_to_anat` function, which was used to co-register the fMRI to the T1w image.

    **Parameters:**
    - fmri_file: Path to the fMRI image with skull.
    - t1w_file: Path to the T1w image with skull.
    - func2struct_mat: Path to the fMRI-to-T1w affine matrix obtained from `fsl_coregister_func_to_anat` function.
    - output_prefix: Full path prefix (including directory) for the files.

    **Outputs:**
    - {output_prefix}_func_mni.nii.gz: fMRI in MNI space.
    - {output_prefix}_mean_func_mni.nii.gz: Mean fMRI in MNI space.
    - {output_prefix}_mean_func_mni_brain_mask.nii.gz: Brain mask of the mean fMRI in MNI space, used for quality control.
    """

    # Standard FSL MNI templates
    mni_brain = "Path/tools/mni_template/mni_template_2mm/brain.nii.gz"
    mni_head  = "Path/tools/mni_template/mni_template_2mm/with_skull.nii.gz"

    # Outputs (make extensions explicit to avoid ambiguity)
    t1w_brain = f"{output_prefix}_t1w_brain.nii.gz"
    # BET -m will produce: <out>_mask (keeping extension)
    t1w_brain_mask = f"{output_prefix}_t1w_brain_mask.nii.gz"

    # func2struct_mat = f"{output_prefix}_func2struct.mat"
    struct2mni_affine = f"{output_prefix}_struct2mni_affine.mat"
    struct2mni_warp = f"{output_prefix}_struct2mni_warp.nii.gz"

    func_mni = f"{output_prefix}_func_mni.nii.gz"  # unmasked
    subj_mask_mni = f"{output_prefix}_subjmask_mni.nii.gz"
    subj_mask_mni_bin = f"{output_prefix}_subjmask_mni_bin_mask.nii.gz"

    func_mni_brain = f"{output_prefix}_func_mni_brain.nii.gz"
    mean_func_mni_brain = f"{output_prefix}_mean_func_mni_brain.nii.gz"

    mean_func_mni = f"{output_prefix}_mean_func_mni.nii.gz"
    mean_fmri_brain = f"{output_prefix}_mean_func_mni_brain.nii.gz"
    mean_fmri_brain_mask = f"{output_prefix}_mean_func_mni_brain_mask.nii.gz"


    out_dir = os.path.dirname(output_prefix)

    cmds = [
        f"mkdir -p {out_dir}",

        f"bet {t1w_file} {t1w_brain} -R -f 0.3 -m",


        f"flirt -ref {mni_brain} -in {t1w_brain} -omat {struct2mni_affine} -dof 12",

        f"fnirt --in={t1w_file} --aff={struct2mni_affine} --cout={struct2mni_warp} --config=T1_2_MNI152_2mm",

        f"applywarp --ref={mni_head} --in={fmri_file} --warp={struct2mni_warp} "
        f"--premat={func2struct_mat} --out={func_mni}",

        f"fslmaths {func_mni} -Tmean {mean_func_mni}",

        f"bet {mean_func_mni} {mean_fmri_brain} -f 0.4 -m",


    ]


    logs = []
    for cmd in cmds:
        log = run_command(cmd)
        logs.append(log)
        # Check if the log indicates an error; if so, return the error message immediately.
        if ("returned non-zero exit code" in log) or ("Failed to run command" in log):
            return log


    return "\n\n".join(logs)





#@tool
# need desc
def fsl_spatial_smooth(input_file: str, output_file: str, fwhm: float = 2.0) -> str:
    """
    Apply spatial smoothing to an image using FSL fslmaths.

    **Parameters:**
    - input_file: Path to the input fMRI image.
    - output_file: Path to save the smoothed image.
    - fwhm: Full-width at half maximum in millimeters (default: 2.0mm).

    **Outputs:**
    - Smoothed image using the given FWHM.
    """
    # FWHM = 2.3548 * sigma ⇒ sigma ≈ fwhm / 2.3548

    sigma = round(fwhm / 2.3548, 2)
    out_dir = os.path.dirname(output_file)
    command = f"mkdir -p {out_dir} && fslmaths {input_file} -s {sigma} {output_file}"
    return run_command(command)




# use this cautiously, I cannot find much information about this command from google, may have problems
#@tool
# need desc
def fsl_temporal_filter(
    input_file: str,
    output_file: str,
    TR: float = 2,
    highpass_freq: Optional[float] = 0.01,
    lowpass_freq: Optional[float] = 0.1
) -> str:
    """
    Apply temporal band-pass filtering to a 4D fMRI image using FSL fslmaths -bptf.

    **Parameters:**
    - input_file: Path to the input fMRI image.
    - output_file: Path to save the temporally filtered fmri image.
    - TR: Repetition time (TR) in seconds.
    - highpass_freq: High-pass frequency threshold, frequeny higher than this value will pass.
    - lowpass_freq: Low-pass frequency threshold, frequeny lower than this value will pass.

    **Outputs:**
    - Band-pass filtered fMRI time series.
    """
    hp_sigma = 0
    lp_sigma = 0

    highpass_sec = 1 / highpass_freq
    lowpass_sec = 1 / lowpass_freq

    if highpass_sec:
        hp_sigma = round(highpass_sec / (2 * TR), 2)
    if lowpass_sec:
        lp_sigma = round(lowpass_sec / (2 * TR), 2)

    out_dir = os.path.dirname(output_file)
    command = f"mkdir -p {out_dir} && fslmaths {input_file} -bptf {hp_sigma} {lp_sigma} {output_file}"
    
    return run_command(command)


