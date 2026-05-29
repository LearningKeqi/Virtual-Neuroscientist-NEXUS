# Guideline: T1w Gray Matter Probability Map-based Prediction

## Overview

This guide outlines the guidance for T1w Gray Matter (GM) Probability Map-based prediction modeling.

## 1. Task Overview

You are given a preprocessed and Quality Control Passed structural MRI training dataset and asked to build a model on T1w gray matter probability maps. The final trained model(s) will be used to perform inference on a hidden test set (which is invisible to you), and higher prediction performance on the hidden test set is preferred. The overall pipeline you can follow is as follows:

1. Resample each subject's GM probability map to a common GM mask if needed, then extract masked voxel-wise features or masked 3D volumes.
2. Train one or more models using leakage-free feature engineering and cross-validation.
3. Submit the trained model or ensemble models for evaluation on a held-out test set.

**Evaluation**: AUROC, F1 (with AD as positive), and Accuracy.

---

## 2. Data Representation and General Principles

You will be provided with the following training data only (no test set data):
1. **Imaging Data:**
   - Format: One `.nii.gz` file per subject.
   - Status: Passed Quality Control (QC) and normalized to the MNI 152 standard space.
   - Content: T1w Gray Matter Probability Map (voxel values typically ranging from 0 to 1).
2. **Label & Demographic Data:**
   - Format: Tabular file (e.g., `.csv` or `.tsv`).
   - Fields: `participant_id`, `diagnosis` (CN or AD).
3. **Gray Matter Mask (MNI GM Mask):**
   - Path: Path/tools/mni_template/mni_template_1mm_aligned/mni152_brain_seg_GM_mask.nii.gz
   - Format: A global `.nii.gz` mask file (in MNI 152 space).
   - Purpose: Used to extract the effective voxel features of the brain's gray matter regions for each subject.

T1w GM probability maps are high-dimensional voxel-wise measurements. After masking, the 
number of voxel features is often far larger than the number of subjects. This is a classic 
high-dimension, small-sample regime, so conservative modeling choices usually generalize 
better than highly flexible models.

- **Voxel-wise feature vectors** are the primary representation for classical ML. They are extracted by masking the 3D GM probability map and flattening the retained voxels into a 1D vector.
- **Masked 3D volumes** are an optional secondary representation for 3D CNNs. They preserve spatial layout but usually require more data and more careful regularization.

**General principle**: prefer simpler, strongly regularized models first; only move to 
higher-capacity models if CV results are stable and clearly better.

---

## 3. Image Alignment and Feature Extraction

### 3.1 Resampling to the GM Mask

Although each subject's GM probability map is already in MNI space, its shape and affine may still differ from the provided GM mask. Before masking:

1. Load the subject image and the GM mask.
2. Check whether the shape and affine match.
3. If they do not match, resample the subject image to the mask space using continuous interpolation.

This step is mandatory. Otherwise, voxel indices will not correspond across subjects.

### 3.2 Masked Feature Extraction

After alignment:

- Apply the GM mask to extract voxel-wise GM probabilities.
- Use the resulting masked voxel vector as the main feature representation for classical models.
- Optionally keep the masked 3D volume for 3D CNN experiments.

### 3.3 Caching

Feature extraction from `.nii.gz` files is relatively slow. Before repeated CV or model search, precompute and cache subject-level masked feature vectors whenever possible, so hyperparameter tuning operates on cached numeric matrices rather than repeatedly re-reading and re-masking the same images.

---

## 4. Model Selection

You should try multiple model families and compare their cross-validated performance before selecting the final model. Different models have different strengths depending on dataset characteristics (sample size, feature dimensionality, class balance, etc.). Use CV results to make evidence-based model selection decisions. Do not assume any model family is inherently superior — the best model is the one that demonstrates the most balanced and stable performance on your specific data.

### 4.1 Classical ML Models

Classical models operate on vectorized voxel-wise features. They are simple, yet efficient, and often provide strong baselines.

- **Regularized Logistic Regression**: A reliable starting point for GM probability map classification. Prefer stronger regularization first, especially when the retained voxel count is still high after feature selection.
- **SVM**: Another strong baseline for high-dimensional neuroimaging data. Linear SVM is usually the safer choice; RBF SVM can help if feature dimensionality has already been reduced substantially, but it is more prone to unstable CV estimates.
- **ElasticNet Logistic Regression**: Useful when you want both shrinkage and sparsity. Can work well after masking when many voxels are weakly informative and only a subset carries signal.
- **XGBoost**: A strong model. If you use it, first reduce dimensionality aggressively by feature selection or feature extraction and keep tree depth small.

### 4.2 3D CNN

3D CNNs can operate directly on masked volumes and preserve spatial structure. However, they usually require more data than classical ML and are more sensitive to optimization choices. Their performance should always be validated through rigorous cross-validation and compared against simpler baselines.

**Hyperparameter guidance for 3D CNN**:
- Start with conservative, well-regularized configurations. Keep base channels small (e.g., 4 or 8), use visible dropout (e.g., 0.3–0.5), and apply weight decay (e.g., 1e-4 to 1e-3).
- Start with **15–30 epochs**. Monitor validation performance after every epoch for checkpoint selection.
- Use small batch sizes (e.g., 2 or 4) when sample size is limited.
- During both CV training and the final full-data retraining stage, evaluate on a validation set after every epoch. Do **not** select the final checkpoint by the last epoch or by training loss alone.
- For the final retraining stage, reserve a **20% stratified validation split** from the full training set for checkpoint selection. Apply **early stopping** if validation performance has not improved for several consecutive epochs.

### 4.3 Recommended Exploration Strategy

Rather than committing to a single model upfront, use cross-validation to compare at least three different configurations. Examples of useful comparisons:
- Regularized Logistic Regression vs. Linear SVM (both with the same feature selection pipeline)
- Same model with different feature_keep_ratio values (e.g., 0.02 vs. 0.05)
- Classical ML baseline vs. 3D CNN
- Same model with vs. without PCA as an additional dimensionality reduction step

These comparisons help you make informed decisions about which configuration generalizes best. The relationship between model complexity and generalization depends on the specific dataset — sometimes simpler models generalize better, sometimes more complex models do. Let the CV results guide your decision.


---

## 5. Feature Engineering Guidance

### 5.1 Leakage-Free Feature Selection

Any feature engineering that uses information from labels or from the feature distribution must be performed inside the training fold only. This includes:

- variance filtering
- univariate feature selection (for example, t-test or ANOVA)
- PCA
- standardization

Never fit these transformations on the full dataset before CV.

### 5.2 Recommended Classical Pipeline

A practical classical pipeline is:

1. Variance filtering to remove constant voxels.
2. Feature selection to keep only a small percentage of the most informative voxels.
3. Standardization when the model is scale-sensitive.
4. A regularized classifier.

---

## 6. Training Stability

Model training on structural MRI datasets can be challenging. Be aware of the following common failure patterns:

- **Overfitting after weak feature reduction**: keeping too many voxels allows the classifier to memorize training folds.
- **Metric imbalance**: a configuration may show acceptable AUROC but weak F1 or ACC due to poor thresholding or skewed predictions.

### 6.1 Checkpoint Selection for 3D CNN Models

For 3D CNNs, after CV identifies the final configuration, do a final retraining stage using the full training cohort but reserve a fresh **20% stratified validation split** for checkpoint selection. During this final retraining stage:
- Evaluate the model on the validation split after every epoch.
- **Do not rely on a single metric or a hard-coded formula to select the checkpoint.** Instead, review the per-epoch validation logs holistically: examine AUROC, F1, and ACC together, check whether the model is collapsing (e.g., F1 dropping to zero, ACC at chance level), and select the checkpoint where all three metrics are reasonably high and balanced.
- Apply **early stopping**: if validation performance has not meaningfully improved for several consecutive epochs, stop training to prevent overfitting.
- **Sanity check after refit**: After selecting the final checkpoint, run inference on the full training set. If the model predicts a large majority of training samples as the same class, the model has likely collapsed during refit — discard it and consider alternative configurations.

Do **not** select the final checkpoint by the last epoch or by training loss alone. This requirement does **not** apply to classical machine learning models; for those models, after CV selects the best hyperparameters, simply refit on the full training set.


---

## 7. CV-Based Model Selection

### 7.1 Adaptive Model Selection — The Core Advantage of an Agent

**This is the most important section of this guide.**

The core advantage of an agent over a static script is the ability to **adaptively reason about training results**. Use this advantage fully.

**Critical workflow requirement**: Training scripts should save all candidate models and output complete CV results (per-fold metrics, training logs, etc.), but should **NOT** automatically select the final model inside the script (e.g., by hard-coding a comparison like `if model_A_auroc > model_B_auroc: select A`). Instead, the final model selection decision must be made by you — the agent — after reviewing all training outputs. This means:
- Your training scripts should produce and save all candidate models along with their full CV reports.
- After the scripts finish, you should read the output logs and CV results, reason about them holistically, and then explicitly decide which model to use as the final submission.
- Do not delegate this decision to a hard-coded rule in the training script. This is where your adaptive reasoning ability matters most.

When making the final model selection, you must holistically review:

1. **Per-fold results**: Examine AUROC, F1, and ACC for every fold individually. A model with consistent, balanced performance across all folds is far more trustworthy than one with high average but erratic per-fold behavior.
2. **Balance across metrics**: All three metrics (AUROC, F1, ACC) should be reasonably high. If one metric is substantially higher than the others, this is a warning sign — not a strength. For example:
   - High AUROC but low F1/ACC → the model ranks subjects well but cannot make reliable binary decisions.
   - High F1 but low ACC → the model may be predicting most subjects as positive, inflating F1 through high sensitivity while specificity is near zero.
   - High ACC but low F1 → the model may be biased toward the majority class.
3. **Training stability**: Review training logs for signs of instability (F1 oscillation, loss divergence, class collapse at certain epochs). Stable training is a prerequisite for reliable generalization.
4. **Model complexity vs. data characteristics**: Consider whether the model complexity is appropriate for the dataset. A complex model that barely outperforms a simple baseline on CV is likely overfitting and will underperform on the test set.

**Your final model selection rationale should explicitly address all of the above factors.** Do not simply pick the configuration with the highest average of any metric or formula. Instead, reason about which configuration is most likely to generalize well to unseen data, considering all available evidence from the training process.

---

## 8. Other Guidance

1. `Resampling is mandatory when needed`: Even if all data are in MNI space, do not assume voxel-wise correspondence unless shape and affine match the GM mask.
2. `Cache extracted features`: Save masked voxel vectors and masked volumes so repeated experiments are efficient and reproducible.
3. **GPU availability**: The current server **has GPUs available**. Do NOT run `nvidia-smi` or any GPU detection command to check — it will incorrectly report no devices on this server, but GPUs are present and functional. Use `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` in your training scripts. When submitting scripts via `run_python_script`, set `need_gpu=True` for any job that requires GPU (e.g., deep learning training). Set `need_gpu=False` for jobs that do not need GPU (e.g., classical ML, data preprocessing).
4. `Final Output & Inference Interface`: You need to output a complete, reproducible codebase, including **Training Script(s)** and a single **Inference Script (`predict.py`)**. Given a list of preprocessed `.nii.gz` GM probability map file paths, `predict.py` script should output predicted probabilities and classification results. Note that the `predict.py` script should clearly specify how to run it, as well as its expected input and output. Providing `predict.py` is mandatory. If using an ensemble, `predict.py` must load all ensemble member models and average their predictions.
5. The above guidelines are intended as general guidance. You can adapt your decisions based on observed CV behavior and training logs. The ultimate goal is to achieve the best possible performance on the hidden test set while preserving leakage-free evaluation.
6. Treat analysis task as an iterative process rather than a one-shot fit: review the validation results or training logs, use them to guide evidence-based refinements (for example, changing feature engineering, changing hyperparameters or trying a different model etc.) You should always make adaptive modeling or refinement decisions based on the run-time observations. Do not defer potential feasible attempts to "next steps" if they can be tried within the current training process; instead, try several feasible modeling designs and refinements before returning final modeling results to the supervisor.
