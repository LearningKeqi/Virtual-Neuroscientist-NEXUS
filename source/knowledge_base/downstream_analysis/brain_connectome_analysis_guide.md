# Guideline: Functional Brain Connectome Analysis

## Overview

This guide outlines the guidance for Functional Brain Connectome Analysis.

## 1. Task Overview

You are given a preprocessed fMRI training dataset and asked to build a model based on resting-state functional connectivity data. The final trained model(s) will be used to perform inference on a hidden test set (which is invisible to you), and higher prediction performance on the hidden test set is preferred. The overall pipeline you can follow is as follows:

1. Construct functional connectivity matrices from fMRI using one or more brain atlases.
2. Train model(s) on the connectivity matrices.
3. Submit the trained model  for evaluation on a held-out test set.

**Evaluation**: AUROC, F1 (with the disorder class as positive), Accuracy. 

---

## 2. Atlas Selection

Available atlases differ in the number of ROIs (brain regions). This directly determines the dimensionality of the connectivity matrix.

- **Low-ROI atlases** (e.g., AAL ~90 ROIs) produce a more compact feature space. CV results tend to be more stable and transfer more reliably to unseen data, especially when sample size is limited relative to the feature space.
- **High-ROI atlases** (e.g., HCP360 ~360 ROIs) capture finer-grained brain network structure but introduce much higher dimensionality. High-ROI atlases may fit better when the number of datasets is large.
- **Medium-ROI atlases** (e.g., Schaefer200 ~200 ROIs) lie between low-ROI and high-ROI atlases in terms of granularity and dimensionality.

**General principle**: The choice of atlas should be informed by the dataset size and the model family. When sample size is small relative to the feature space, prefer lower-dimensional representations. When sample size is large, higher-ROI atlases may capture richer information. Always evaluate multiple atlases via cross-validation rather than committing to one upfront.

---

## 3. Feature Engineering

### 3.1 Connectivity Construction
Use `construct_functional_brain_connectivity` to generate a connectivity matrix for each subject (shape `num_ROIs × num_ROIs`). All subject-level connectivity matrices should be combined into a single `.npy` file and saved in `workspace/data/`.



### 3.2 Feature Representations for Different Model Families
- **Deep learning models (BNT, NeuroGraph, etc.)**: Use the full connectivity matrix (R × R) as the node feature matrix (connection profile as node feature). No vectorization needed.
- **Classical ML**: Vectorize the upper triangle of the connectivity matrix. Optionally apply PCA for dimensionality reduction or t-test feature selection.
- **Sparsified Connectivity**: Only needed for NeuroGraph adjacency matrix construction. Use `sparsify_connectivity_matrices` with default `keep_ratio=0.1`.

---

## 4. Model Selection

You should try multiple model families and compare their cross-validated performance before selecting the final model. Different model families have different inductive biases, and the best choice depends on the specific dataset characteristics (sample size, number of ROIs, class balance, site heterogeneity, etc.). Use CV results to make evidence-based model selection decisions. Do not assume any model family is inherently superior — the best model is the one that demonstrates the most balanced and stable performance on your specific data.

### 4.1 Deep Learning Models (BNT, NeuroGraph, etc.)

State-of-the-art Deep learning models such as BrainNetworkTransformer (BNT) and NeuroGraph are designed to capture graph-level structure in brain connectivity matrices. They can be powerful when sufficient data is available, but may also be prone to overfitting or training instability depending on dataset size and characteristics. Their performance should always be validated through rigorous cross-validation and compared against simpler baselines.


### 4.2 Classical ML (e.g., Logistic Regression, SVM, ElasticNet, Random Forest)

Classical models operate on vectorized connectivity features (the upper triangle of the connectivity matrix). They are simple but efficient, and often provide strong baselines. Well-regularized classical models can be highly competitive, especially when sample size is limited relative to the feature space. For small sample sizes (<300 subjects), well-regularized classical models with proper feature engineering may outperform deep learning models in generalization.

### 4.3 Recommended Exploration Strategy

Rather than committing to a single model family upfront, use cross-validation to compare several different configurations (different atlas & feature engineering & models & hyperparameters).

These comparisons help you make informed decisions about which configuration generalizes best, rather than relying on assumptions. The relationship between model complexity and generalization depends on the specific dataset — sometimes simpler models generalize better, sometimes more complex models do. Let the CV results guide your decision.


---

## 5. Training Stability

Model training on brain-imaging datasets can be challenging. Be aware of the following common failure patterns:

- **F1 oscillation**: F1 swings between high and zero within consecutive epochs as the model alternates between predicting most-positive and most-negative.
- **Class collapse**: ACC stuck at ~0.50, indicating the model is predicting the same class for nearly all subjects.

### 5.1 Checkpoint Selection for Deep Learning Models

For deep learning models, after CV identifies the final configuration, do a final retraining stage using the full training cohort but reserve a fresh **20% stratified validation split** for checkpoint selection. During this final retraining stage:
- Evaluate the model on the validation split after every epoch.
- **Do not rely on a single metric or a hard-coded formula to select the checkpoint.** Instead, review the per-epoch validation logs holistically: examine AUROC, F1, and ACC together, check whether the model is collapsing (e.g., F1 dropping to zero, ACC at chance level), and select the checkpoint where all three metrics are reasonably high and balanced.
- Apply **early stopping**: if validation performance has not meaningfully improved for several consecutive epochs, stop training to prevent overfitting.

Do **not** select the final deep-learning checkpoint by the last epoch or by training loss alone. This requirement does **not** apply to classical machine learning models (e.g., Logistic Regression, SVM, ElasticNet, Random Forest); for those models, after CV selects the best hyperparameters, simply refit on the full training set.

---

## 6. Cross-Validation Strategy

### 6.1 Standard CV
Use k-fold StratifiedKFold (e.g., 5-fold) as the default cross-validation strategy. Report AUROC, F1, and ACC for each fold. Use the per-fold validation results as the basis for model comparison — but compare holistically across all metrics and folds, not by averaging a single metric.

---

## 7. Iterative Refinement Protocol

This section defines the iterative process you should follow. Do NOT treat modeling as a one-shot task.

### 7.1 Iterative Workflow
After each training run, review the cross-validated results and use them to guide your next action:
1. **Run several initial configurations** with different model families, atlases, or hyperparameters.
2. **Evaluate CV results**. Check for collapsed models, high variance across folds, metric imbalance, or poor overall performance.
3. **Decide on potential refinements** based on what you observed.
4. **Run the refined configurations** and compare to previous results.
5. **Repeat** until you have tried at least two meaningfully different configurations, observe that CV results do not trigger any of the conditions in Section 8.2, and you are confident in your final selection.

### 7.2 Refinement Triggers

you should also trigger further refinement if:
- **No configuration yet achieves balanced, reasonably good performance across all three metrics.** If all tried configurations have at least one metric that is weak or substantially lower than the others, you have not yet found a good candidate — keep exploring.
- **All configurations show high variance across folds.** This suggests the modeling approach or feature representation may be fundamentally mismatched with the data. Try a different atlas, feature engineering, or model family.
- **A complex model only marginally outperforms a simpler one.** The simpler model is likely to generalize better.


### 7.3 What Counts as a Meaningful Refinement
Each of the following counts as a meaningful refinement attempt:
- Switching model family
- Switching atlas
- Changing hyperparameters based on observed CV results
- Changing feature engineering
- Changing CV strategy


### 7.4 Reporting Format
When returning results to the supervisor, include:
1. A comparison table of ALL configurations you tried, with columns: Model, Atlas, Feature Engineering, CV Strategy, AUROC (mean±std), F1 (mean±std), ACC (mean±std).
2. Your recommendation for the best configuration and the reasoning behind it — this reasoning must address metric balance, per-fold stability, and training behavior, not just average metric values.
3. The saved model paths and inference script path.

---

## 8. Other Guidance

1. The above guidelines are intended as general guidance. You can adapt your decisions based on the behaviors or the training logs you observe during model training process. The ultimate goal is to achieve the best possible performance on the hidden test set.
2. **GPU availability**: The current server **has GPUs available**. Do NOT run `nvidia-smi` or any GPU detection command to check — it will incorrectly report no devices on this server, but GPUs are present and functional. Use `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` in your training scripts. When submitting scripts via `run_python_script`, set `need_gpu=True` for any job that requires GPU (e.g., deep learning training). You must set `need_gpu=False` for jobs that do not need GPU (e.g., classical ML, data preprocessing).
3. `Final Output & Inference Interface`: You need to output a complete, reproducible codebase, including **Training Script(s)** and a single **Inference Script (`predict.py`)**. Given a list of preprocessed `.nii.gz` fMRI file paths, `predict.py` script should output predicted probabilities and classification results. Note that the `predict.py` script should clearly specify how to run it, as well as its expected input and output. Providing `predict.py` is mandatory. If using an ensemble, `predict.py` must load all ensemble member models and average their predictions.
4. Treat analysis task as an iterative process rather than a one-shot fit: review the validation results or training logs, use them to guide evidence-based refinements (for example, changing feature engineering, changing hyperparameters or trying a different model etc.) You should always make adaptive modeling or refinement decisions based on the run-time observations. Do not defer potential feasible attempts to "next steps" if they can be tried within the current training process; instead, try several feasible modeling designs and refinements before returning final modeling results to the supervisor.
5. **Prefer targeted exploration over exhaustive search**: When selecting or refining models, do not expand the search space mechanically into a large Cartesian-product grid. Start with a small set of diverse, representative configurations, use the observed validation results to identify the most promising direction, and only then explore a narrower set of refinements around that direction. The purpose of iterative refinement is to make better decisions from evidence, not to maximize the number of settings tried.
