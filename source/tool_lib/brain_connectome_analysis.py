
# this file stores the codebase for the downstream analysis which agent could directly import in the python script
# need desc
def construct_functional_brain_connectivity(fmri_img_path: str, parcellation_atlas, method='correlation'):
    """
    Construct the functional brain connectivity from a single-subject fMRI image file
    and the parcellation atlas.

    **Parameters:**
    - fmri_img_path: (str), the path of the fMRI NIfTI image file (.nii or .nii.gz) for a single subject,
      with shape (W, H, D, T), where W,H,D are the spatial dimensions, and T is the time dimension.
    - parcellation_atlas: (str), choose from ['AAL90', 'Schaefer200', 'HCP360'], the parcellation atlas, number of ROIs is 90, 200, 360 respectively.
    - method: (str), choose from ['correlation', 'causality'], default is 'correlation'.

    **Outputs:**
    - (numpy array) The functional brain connectivity. with shape (num_ROIs, num_ROIs).
    """

    import numpy as np
    import nibabel as nib
    from nilearn import image as nimg

    # --- method selection ---
    if method not in ("correlation", "causality"):
        raise ValueError(f"Unsupported method: {method!r}. Supported: 'correlation', 'causality'.")
    if method == "causality":
        raise NotImplementedError("Causality-based functional connectivity is not implemented yet.")

    # --- atlas path mapping (fill in real paths in your environment) ---
    atlas_path_map = {
        "AAL90": "Path/tools/atlas/aal_wo_cerebellum.nii",
        "Schaefer200": "Path/tools/atlas/Schaefer2018_200Parcels_17Networks_order_FSLMNI152_1mm.nii.gz",
        "HCP360": "Path/tools/atlas/MNI_Glasser_HCP_v1.0.nii",
    }

    if parcellation_atlas not in atlas_path_map:
        raise ValueError(
            f"Unknown parcellation_atlas: {parcellation_atlas!r}. "
            f"Supported options: {list(atlas_path_map.keys())}"
        )

    atlas_path = atlas_path_map[parcellation_atlas]

    # --- load atlas image (3D, int labels; 0 = background) ---
    atlas_img = nimg.load_img(atlas_path)  # (Wa, Ha, Da)

    # --- load fMRI image from path and normalize to (N, W, H, D, T) ---
    fmri_img = nimg.load_img(fmri_img_path)
    fmri_data = np.asarray(fmri_img.get_fdata())

    if fmri_data.ndim != 4:
        raise ValueError(
            f"fMRI image at {fmri_img_path!r} must be 4D (W,H,D,T); got shape {fmri_data.shape}."
        )

    nx, ny, nz, n_time = fmri_data.shape

    fmri_ref_img = nimg.index_img(fmri_img, 0)

    if atlas_img.shape != (nx, ny, nz):
        atlas_img_resampled = nimg.resample_to_img(
            atlas_img,
            fmri_ref_img,
            interpolation="nearest",
        )
    else:
        atlas_img_resampled = atlas_img

    atlas_data = np.asarray(atlas_img_resampled.get_fdata())
    atlas_labels = np.rint(atlas_data).astype(np.int32)

    if atlas_labels.shape != (nx, ny, nz):
        raise RuntimeError(
            f"Resampled atlas shape {atlas_labels.shape} does not match fMRI spatial shape {(nx, ny, nz)}."
        )

    # --- derive ROI time series ---
    roi_labels = np.unique(atlas_labels)
    roi_labels = roi_labels[roi_labels > 0]
    if roi_labels.size == 0:
        raise ValueError("No non-zero ROI labels found in the parcellation atlas.")

    n_rois = int(roi_labels.size)

    atlas_flat = atlas_labels.ravel()
    roi_voxel_indices = [np.where(atlas_flat == roi_id)[0] for roi_id in roi_labels]

    empty_rois = [int(roi_id) for roi_id, idx in zip(roi_labels, roi_voxel_indices) if idx.size == 0]
    if empty_rois:
        raise ValueError(
            f"The following ROI labels have no voxels after resampling: {empty_rois}. "
            "Please check your atlas and fMRI alignment."
        )

    subj_data = fmri_data
    vox_by_time = subj_data.reshape(-1, n_time)

    roi_time_series = np.empty((n_rois, n_time), dtype=np.float64)
    for i, idx in enumerate(roi_voxel_indices):
        roi_voxels = vox_by_time[idx, :]  # (n_voxels_i, n_time)
        roi_time_series[i, :] = roi_voxels.mean(axis=0)

    roi_time_series -= np.nanmean(roi_time_series, axis=1, keepdims=True)
    roi_time_series = np.nan_to_num(roi_time_series)

    corr_matrix = np.corrcoef(roi_time_series)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    return corr_matrix.astype(np.float32)


# need desc
def sparsify_connectivity_matrices(orig_connectivity_matrices, keep_ratio, only_positive=True):
    """
    A function to sparsify the brain connectivity matrices.
    It keeps the edges with absolute weights in the top specified keep_ratio and zeros out the rest.
    
    **Parameters:**
    - orig_connectivity_matrices (torch.Tensor): All subjects' original brain connectivity matrices with shape (num_subjects, num_ROIs, num_ROIs).
    - keep_keep_ratio (float): The keep_ratio of edges to keep based on their absolute weight.
    - only_positive (boolean): whether keep only positive correlation. Default is True.

    **Outputs:**
    - (torch.Tensor) The sparsified connectivity matrices with the same shape as orig_connectivity_matrices.
    """

    X = orig_connectivity_matrices.clone()

    negative_edge = False

    if keep_ratio < 0:
        negative_edge = True
        keep_ratio = -1 * keep_ratio
        print(f'negative_edge={negative_edge}, keep_ratio = {keep_ratio}')

    if only_positive:
        if not negative_edge:
            print('only positive correlations')
            X[X<0]=0
        else:
            print('only negative correlations')
            X[X>0]=0


    batch_size, num_nodes, _ = X.size()

    if keep_ratio == 1:
        for i in range(batch_size):
            X[i].fill_diagonal_(1)
        return X


    if keep_ratio == 0:    # keep self-loop
        thresholded_X = torch.zeros_like(X)
        for i in range(batch_size):
            thresholded_X[i].fill_diagonal_(1)
        
        return thresholded_X
    
    # Create a tensor to store the thresholded adjacency matrices
    thresholded_X = torch.zeros_like(X)
    
    for i in range(batch_size):
        if only_positive:
            if not negative_edge:
                # # Flatten the upper triangular part of the matrix to avoid duplicating symmetric edges
                upper_triangular_flat = X[i].triu(diagonal=1).flatten()   # diagnoal = 1, no diagonal
                upper_num_positive = torch.sum(upper_triangular_flat>0)

                # Number of edges to keep per adjacency matrix
                num_edges_to_keep = int(keep_ratio * upper_num_positive)

            else:
                upper_triangular_flat = X[i].triu(diagonal=1).flatten()   # diagnoal = 1, no diagonal
                upper_num_positive = torch.sum(upper_triangular_flat<0)
                num_edges_to_keep = int(keep_ratio * upper_num_positive)
        

        else:
            upper_triangular_flat = X[i].triu(diagonal=1).flatten()
            num_edges_to_keep = int(keep_ratio * num_nodes * (num_nodes - 1) / 2)  # Divide by 2 because the matrices are symmetric



        # Get the absolute values and sort them to find the threshold
        values, indices = torch.abs(upper_triangular_flat).sort(descending=True)
        threshold = values[num_edges_to_keep]
        
        # Apply thresholding
        mask = torch.abs(X[i]) >= threshold
        
        # Apply the symmetrical mask and update the thresholded adjacency matrix
        thresholded_X[i] = X[i] * mask
    

    for i in range(batch_size):
        thresholded_X[i].fill_diagonal_(1) # keep self-loop
        
    return thresholded_X



########################################################################################
########################### Brain Connectome Analysis Models ###########################
########################################################################################


#-------------------------------- Brain Network Transformer --------------------------------

import torch
import torch.nn as nn
from .bnt_comp.ptdec import DEC
from .bnt_comp.components import InterpretableTransformerEncoder

class TransPoolingEncoder(nn.Module):
    """
    Transformer encoder with Pooling mechanism.
    Input size: (batch_size, input_node_num, input_feature_size)
    Output size: (batch_size, output_node_num, input_feature_size)
    """

    def __init__(self, input_feature_size, input_node_num, hidden_size, output_node_num, num_heads, pooling=True, orthogonal=True, freeze_center=False, project_assignment=True):
        super().__init__()
        self.transformer = InterpretableTransformerEncoder(d_model=input_feature_size, nhead=num_heads,
                                                           dim_feedforward=hidden_size,
                                                           batch_first=True)

        self.pooling = pooling
        if pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(input_feature_size *
                          input_node_num, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size,
                          input_feature_size * input_node_num),
            )
            self.dec = DEC(cluster_number=output_node_num, hidden_dimension=input_feature_size, encoder=self.encoder,
                           orthogonal=orthogonal, freeze_center=freeze_center, project_assignment=project_assignment)

    def is_pooling_enabled(self):
        return self.pooling

    def forward(self, x):
        x = self.transformer(x)
        if self.pooling:
            x, assignment = self.dec(x)
            return x, assignment
        return x, None

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()

    def loss(self, assignment):
        return self.dec.loss(assignment)



# need desc
class BrainNetworkTransformer(nn.Module):
    def __init__(
        self, 
        num_rois,
        num_layers,
        num_clusters,
        hidden_size,
        num_heads,
        pred_task,
        num_classes,
    ):
        """
        One of the state-of-the-art models for functional brain connectome analysis. A graph transformer-based model for brain network analysis.

        **Parameters:**
        - num_rois: (int). number of ROIs.
        - num_layers: (int). number of layers of the model.
        - num_clusters: (int). number of ROIs clusters. Should set as a small number when training on a small dataset.
        - hidden_size: (int). hidden size of FFN in transformer encoder.
        - num_heads: (int). number of heads of transformer encoder. Usually set as 2. Note that the number of heads **must be divisible by the input feature size (number of ROIs).**
        - pred_task: (str). 'classification' or 'regression'.
        - num_classes: (int). number of classes for classification task.
        """

        # 在docstring中可以考虑添加对模型的简介，这个介绍有什么用？让Agent知道这个模型的基本原理，才能知道各参数的意义以及如何设置。

        super().__init__()

        self.attention_list = nn.ModuleList()
        forward_dim = num_rois
        sizes = [num_rois] * num_layers
        sizes[-1] = num_clusters

        in_sizes = [num_rois] + sizes[:-1]
        do_pooling = [False] * len(sizes)
        do_pooling[-1] = True

        self.do_pooling = do_pooling
        for index, size in enumerate(sizes):
            self.attention_list.append(
                TransPoolingEncoder(input_feature_size=forward_dim,
                                    input_node_num=in_sizes[index],
                                    hidden_size=hidden_size,
                                    output_node_num=size,
                                    num_heads=num_heads,
                                    pooling=do_pooling[index],
                                    orthogonal=True,
                                    freeze_center=True,
                                    project_assignment=True))

        self.dim_reduction = nn.Sequential(
            nn.Linear(forward_dim, 8),
            nn.LeakyReLU()
        )
    
        output_dim = num_classes if pred_task == 'classification' else 1


        self.fc = nn.Sequential(
            nn.Linear(8 * sizes[-1], 8),
            nn.LeakyReLU(),
            nn.Linear(8, output_dim)
        )




    def forward(self,
                node_feature: torch.tensor):
        
        """
        Forward pass of the Brain Network Transformer model.

        Parameters:
        - node_feature: (torch.tensor), with shape (batch_size, num_rois, node_feature_size)

        Returns:
        - (torch.tensor) The output logits. Shape: (batch_size, output_dim), e.g., for binary classification, output_dim=2.
        """

        bz, _, _, = node_feature.shape


        assignments = []

        for atten in self.attention_list:
            node_feature, assignment = atten(node_feature)
            assignments.append(assignment)

        node_feature = self.dim_reduction(node_feature)

        node_feature = node_feature.reshape((bz, -1))

        return self.fc(node_feature)
    

    

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]

    def get_cluster_centers(self) -> torch.Tensor:
        """
        Get the cluster centers, as computed by the encoder.

        :return: [number of clusters, hidden dimension] Tensor of dtype float
        """
        return self.dec.get_cluster_centers()

    def loss(self, assignments):
        """
        Compute KL loss for the given assignments. Note that not all encoders contain a pooling layer.
        Inputs: assignments: [batch size, number of clusters]
        Output: KL loss
        """
        decs = list(
            filter(lambda x: x.is_pooling_enabled(), self.attention_list))
        assignments = list(filter(lambda x: x is not None, assignments))
        loss_all = None

        for index, assignment in enumerate(assignments):
            if loss_all is None:
                loss_all = decs[index].loss(assignment)
            else:
                loss_all += decs[index].loss(assignment)
        return loss_all



#-------------------------------- NeuroGraph --------------------------------


import torch
from torch.nn import Linear
from torch import nn
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import aggr
import torch.nn.functional as F
from torch_geometric.nn import APPNP, MLP, GCNConv, GINConv, SAGEConv, GraphConv, TransformerConv, ChebConv, GATConv, SGConv, GeneralConv
from torch.nn import Conv1d, MaxPool1d, ModuleList
import random
import numpy as np

softmax = torch.nn.LogSoftmax(dim=1)



class NeuroGraph(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        num_rois,
        num_layers,
        hidden,
        task,
    ):
        """
        NeuroGraph model. A graph neural network model for brain network analysis with residual connection.

        **Parameters:**
        - hidden_channels: (int). Number of hidden channels used by each graph convolution layer.
        - num_rois: (int). Number of brain regions of interest (ROIs).
        - num_layers: (int). Number of graph convolution layers in the model.
        - hidden: (int). Hidden dimension used by the downstream MLP classifier/regressor.
        - task: (str). Task type, usually `'classification'` or `'regression'`.

        """

        super(NeuroGraph, self).__init__()
        self.has_mid_layer = True
        self.has_residual = True
        self.has_self_loop = False

        self.convs = ModuleList()
        self.aggr = aggr.MeanAggregation()

        if num_layers>0:
            self.convs.append(GCNConv(num_rois, hidden_channels))
            for i in range(0, num_layers - 1):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        input_dim1 = int(((num_rois * num_rois)/2)- (num_rois/2)+(hidden_channels*num_layers))
        input_dim = int(((num_rois * num_rois)/2)- (num_rois/2))
        self.bn = nn.BatchNorm1d(input_dim)

        if self.has_mid_layer:
            self.bnh = nn.BatchNorm1d(hidden_channels*num_layers)
        else:
            self.bnh = nn.BatchNorm1d(hidden_channels)

        output_dim = 2 if task == 'classification' else 1

        if not self.has_residual:
            if self.has_mid_layer:
                input_dim1 = int(hidden_channels*num_layers)
            else:
                input_dim1 = hidden_channels

            
        self.mlp = nn.Sequential(
            nn.Linear(input_dim1, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden//2, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear((hidden//2), output_dim),
        )

    def forward(self, m, node_feature):
        """
        Forward pass of the NeuroGraph model.

        Parameters:
        - m: (torch.tensor), with shape (batch_size, num_rois, num_rois). The adjacency matrix of the brain graph generated by `sparsify_connectivity_matrices`.
        - node_feature: (torch.tensor), with shape (batch_size, num_rois, node_feature_size)

        Returns:
        - (torch.tensor) The output logits. Shape: (batch_size, output_dim), e.g., for binary classification, output_dim=2.
        """


        num_graphs, num_nodes, _ = m.shape

        x, edge_index, batch = self.transform_data(m, node_feature)

        # print(f'edge_index.len={edge_index.shape}')

        xs = [x]        
        for conv in self.convs:
            xs += [conv(xs[-1], edge_index).tanh()]
        h = []
        for i, xx in enumerate(xs):
            if i== 0:
                xx = xx.reshape(num_graphs, x.shape[1],-1)
                # x = torch.stack([t.triu(diagonal=1).flatten()[t.triu(diagonal=1).flatten().nonzero(as_tuple=True)] for t in xx])
                offset = 0 if self.has_self_loop else 1
                rows, cols = torch.triu_indices(row=num_nodes, col=num_nodes, offset=offset)  # offset 0 including diag
                x = xx[:, rows, cols]

                x = self.bn(x)
            else:
                xx = self.aggr(xx,batch)
                h.append(xx)
        
        if self.has_mid_layer:
            h = torch.cat(h,dim=1)   
        else:
            h = h[-1]
        
        h = self.bnh(h)


        if self.has_residual:
            x = torch.cat((x,h),dim=1)
        else:
            x = h
            
        x = self.mlp(x)
        
        return x


    def transform_data(self, m, node_feature):
        '''
        Transform input data 'm' and 'node_feature' into the forms needed by Braingnn model.
        Input: 
                'm' - adjacency matrix generated before. [batch_size, num_nodes, num_nodes]   
                'node_feature' - node feature generated before. [batch_size, num_nodes, node_feature]
        Output: 
                'x' - node feature. [batch_num_nodes, node_feature]
                'edge_index' - each column represents an edge. [2, batch_num_edges]
                'batch' - a column vector which maps each node to its respective graph in the batch. [batch_num_nodes, 1]
                'edge_attr' - edge weights. [batch_num_edges, 1]
                'pos' - one-hot regional information. Its ROI representation ri is a N-dimensional vector with 1 in the i th entry and 0 for the other entries. [batch_num_nodes, num_nodes]

        '''

        ## handling x
        x = node_feature.view(-1, node_feature.size(2))

        ## handling edge_index and edge_attr
        bz = m.shape[0]
        num_nodes = m.shape[1]
        all_edge_indices = []
        # all_edge_weights = []
        
        for b in range(bz):
            row, col = torch.where(m[b] != 0)
            row += b * num_nodes
            col += b * num_nodes

            all_edge_indices.append(torch.stack([row, col], dim=0))
            # all_edge_weights.append(m[b, m[b] != 0])

        edge_index = torch.cat(all_edge_indices, dim=1)
        # edge_attr = torch.cat(all_edge_weights)

        ## handling batch 
        batch = torch.arange(bz).repeat_interleave(num_nodes).view(-1, 1).squeeze()

        return x.cuda(), edge_index.cuda(), batch.cuda()




#-------------------------------- MLP --------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MLP_Connectome(torch.nn.Module):
    def __init__(self, num_mlp_layer, num_rois, task):
        """
        MLP_Connectome model. A multilayer perceptron model for brain connectome analysis.

        **Parameters:**
        - num_mlp_layer: (int). Number of MLP layers applied to node features.
        - num_rois: (int). Number of brain regions of interest (ROIs).
        - task: (str). Task type, usually `'classification'` or `'regression'`.

        """
        super(MLP_Connectome, self).__init__()
        print(f'Using MLP Node model')

        self.mlp_list = nn.ModuleList()
        # self.norms = nn.ModuleList()
        self.dropout= nn.Dropout(0.2)
        for i in range(num_mlp_layer):
            self.mlp_list.append(nn.Linear(num_rois, num_rois))
            # self.norms.append(nn.LayerNorm(num_rois))


        if task == 'classification':
            last_output_dim = 2
        elif task == 'regression':
            last_output_dim = 1

        fc_input_dim = num_rois * num_rois

        self.fc = nn.Sequential(nn.Linear(fc_input_dim, last_output_dim, bias=True))


    def forward(self, node_feature):
        """
        Forward pass of the MLP_Connectome model.

        Parameters:
        - node_feature: (torch.tensor), with shape (batch_size, num_rois, node_feature_size), node_feature is usually the connection profile.

        Returns:
        - (torch.tensor) The output logits. Shape: (batch_size, output_dim), e.g., for binary classification, output_dim=2.
        """
        mlp_feature = node_feature.clone()
        for idx, mlp_layer in enumerate(self.mlp_list):
            mlp_feature = mlp_layer(mlp_feature)
            # mlp_feature = self.norms[idx](mlp_feature)
            mlp_feature = F.relu(mlp_feature)
            mlp_feature = self.dropout(mlp_feature)

        # flatten
        bz = mlp_feature.shape[0]
        mlp_feature = mlp_feature.reshape((bz, -1))

        return self.fc(mlp_feature)



# Classic Machine Learning Models to be implemented


class _BaseConnectomeSKLearnModel:
    def __init__(
        self,
        task,
        feature_keep_ratio=1.0,
        include_diagonal=False,
        use_scaler=True,
        random_state=42,
        **model_kwargs,
    ):
        self.task = task
        self.feature_keep_ratio = feature_keep_ratio
        self.include_diagonal = include_diagonal
        self.use_scaler = use_scaler
        self.random_state = random_state
        self.model_kwargs = model_kwargs

        self.selector = None
        self.scaler = None
        self.model = None

    def _to_numpy_connectivity_matrices(self, connectivity_matrices):
        if torch.is_tensor(connectivity_matrices):
            connectivity_matrices = connectivity_matrices.detach().cpu().numpy()

        connectivity_matrices = np.asarray(connectivity_matrices)

        if connectivity_matrices.ndim == 2:
            if connectivity_matrices.shape[0] != connectivity_matrices.shape[1]:
                raise ValueError(
                    "A single connectivity matrix must be square with shape (num_rois, num_rois)."
                )
            connectivity_matrices = connectivity_matrices[None, ...]

        if connectivity_matrices.ndim != 3:
            raise ValueError(
                "connectivity_matrices must have shape (num_subjects, num_rois, num_rois)."
            )

        if connectivity_matrices.shape[1] != connectivity_matrices.shape[2]:
            raise ValueError("Each connectivity matrix must be square.")

        return connectivity_matrices

    def _vectorize_upper_triangle(self, connectivity_matrices):
        connectivity_matrices = self._to_numpy_connectivity_matrices(connectivity_matrices)
        diagonal_offset = 0 if self.include_diagonal else 1
        rows, cols = np.triu_indices(connectivity_matrices.shape[1], k=diagonal_offset)
        return connectivity_matrices[:, rows, cols]

    def _build_selector(self):
        if not (0 < self.feature_keep_ratio <= 1):
            raise ValueError("feature_keep_ratio must be in the interval (0, 1].")

        if self.feature_keep_ratio == 1:
            return None

        percentile = max(1, min(100, int(round(self.feature_keep_ratio * 100))))

        if self.task == "classification":
            from sklearn.feature_selection import SelectPercentile, f_classif

            return SelectPercentile(score_func=f_classif, percentile=percentile)

        if self.task == "regression":
            from sklearn.feature_selection import SelectPercentile, f_regression

            return SelectPercentile(score_func=f_regression, percentile=percentile)

        raise ValueError(f"Unsupported task type: {self.task!r}.")

    def _build_model(self):
        raise NotImplementedError

    def _transform_features(self, connectivity_matrices, fit=False, y=None):
        from sklearn.preprocessing import StandardScaler

        X = self._vectorize_upper_triangle(connectivity_matrices)

        if fit:
            self.selector = self._build_selector()
            if self.selector is not None:
                X = self.selector.fit_transform(X, y)

            self.scaler = StandardScaler() if self.use_scaler else None
            if self.scaler is not None:
                X = self.scaler.fit_transform(X)
            return X

        if self.selector is not None:
            X = self.selector.transform(X)

        if self.scaler is not None:
            X = self.scaler.transform(X)

        return X

    def fit(self, connectivity_matrices, y):
        y = np.asarray(y)
        X = self._transform_features(connectivity_matrices, fit=True, y=y)
        self.model = self._build_model()
        self.model.fit(X, y)
        return self

    def predict(self, connectivity_matrices):
        X = self._transform_features(connectivity_matrices, fit=False)
        return self.model.predict(X)

    def score(self, connectivity_matrices, y):
        X = self._transform_features(connectivity_matrices, fit=False)
        return self.model.score(X, y)

    def predict_proba(self, connectivity_matrices):
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError(f"{self.__class__.__name__} does not support predict_proba().")

        X = self._transform_features(connectivity_matrices, fit=False)
        return self.model.predict_proba(X)

    def decision_function(self, connectivity_matrices):
        if not hasattr(self.model, "decision_function"):
            raise AttributeError(f"{self.__class__.__name__} does not support decision_function().")

        X = self._transform_features(connectivity_matrices, fit=False)
        return self.model.decision_function(X)

    def get_selected_feature_mask(self):
        if self.selector is None:
            return None
        return self.selector.get_support()


class LogReg_Connectome(_BaseConnectomeSKLearnModel):
    def __init__(
        self,
        feature_keep_ratio=1.0,
        include_diagonal=False,
        use_scaler=True,
        random_state=42,
        **model_kwargs,
    ):
        """
        Logistic regression model for brain connectome analysis using vectorized upper-triangular connectivity features.

        **Parameters:**
        - feature_keep_ratio: (float). Ratio of features to keep during supervised feature selection, in the interval (0, 1].
        - include_diagonal: (bool). Whether to include diagonal elements when vectorizing the connectivity matrix. Default is False.
        - use_scaler: (bool). Whether to standardize features before model fitting. Default is True.
        - random_state: (int). Random seed used by the sklearn model when applicable.
        - **model_kwargs: Additional keyword arguments forwarded to `sklearn.linear_model.LogisticRegression`.

        """
        model_random_state = model_kwargs.pop("random_state", random_state)

        default_model_kwargs = {
            "max_iter": 1000,
            "solver": "liblinear",
            "random_state": model_random_state,
        }
        default_model_kwargs.update(model_kwargs)

        super().__init__(
            task="classification",
            feature_keep_ratio=feature_keep_ratio,
            include_diagonal=include_diagonal,
            use_scaler=use_scaler,
            random_state=model_random_state,
            **{k: v for k, v in default_model_kwargs.items() if k != "random_state"},
        )
        self.model_kwargs["random_state"] = model_random_state

    def _build_model(self):
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(**self.model_kwargs)


class ElasticNet_Connectome(_BaseConnectomeSKLearnModel):
    def __init__(
        self,
        feature_keep_ratio=1.0,
        include_diagonal=False,
        use_scaler=True,
        random_state=42,
        **model_kwargs,
    ):
        """
        Elastic Net regression model for brain connectome analysis using vectorized upper-triangular connectivity features.

        **Parameters:**
        - feature_keep_ratio: (float). Ratio of features to keep during supervised feature selection, in the interval (0, 1].
        - include_diagonal: (bool). Whether to include diagonal elements when vectorizing the connectivity matrix. Default is False.
        - use_scaler: (bool). Whether to standardize features before model fitting. Default is True.
        - random_state: (int). Random seed used by the sklearn model when applicable.
        - **model_kwargs: Additional keyword arguments forwarded to `sklearn.linear_model.ElasticNet`.

        """
        model_random_state = model_kwargs.pop("random_state", random_state)

        default_model_kwargs = {
            "random_state": model_random_state,
        }
        default_model_kwargs.update(model_kwargs)

        super().__init__(
            task="regression",
            feature_keep_ratio=feature_keep_ratio,
            include_diagonal=include_diagonal,
            use_scaler=use_scaler,
            random_state=model_random_state,
            **{k: v for k, v in default_model_kwargs.items() if k != "random_state"},
        )
        self.model_kwargs["random_state"] = model_random_state

    def _build_model(self):
        from sklearn.linear_model import ElasticNet

        return ElasticNet(**self.model_kwargs)


class SVM_Connectome(_BaseConnectomeSKLearnModel):
    def __init__(
        self,
        task="classification",
        feature_keep_ratio=1.0,
        include_diagonal=False,
        use_scaler=True,
        random_state=42,
        **model_kwargs,
    ):
        """
        Support vector machine model for brain connectome analysis using vectorized upper-triangular connectivity features.

        **Parameters:**
        - task: (str). Task type, choose from `'classification'` or `'regression'`.
        - feature_keep_ratio: (float). Ratio of features to keep during supervised feature selection, in the interval (0, 1].
        - include_diagonal: (bool). Whether to include diagonal elements when vectorizing the connectivity matrix. Default is False.
        - use_scaler: (bool). Whether to standardize features before model fitting. Default is True.
        - random_state: (int). Random seed used by the sklearn model when applicable.
        - **model_kwargs: Additional keyword arguments forwarded to `sklearn.svm.SVC` or `sklearn.svm.SVR`.

        """
        default_model_kwargs = {}
        if task == "classification":
            default_model_kwargs["probability"] = True

        default_model_kwargs.update(model_kwargs)

        super().__init__(
            task=task,
            feature_keep_ratio=feature_keep_ratio,
            include_diagonal=include_diagonal,
            use_scaler=use_scaler,
            random_state=random_state,
            **default_model_kwargs,
        )

    def _build_model(self):
        if self.task == "classification":
            from sklearn.svm import SVC

            return SVC(**self.model_kwargs)

        if self.task == "regression":
            from sklearn.svm import SVR

            return SVR(**self.model_kwargs)

        raise ValueError(f"Unsupported task type: {self.task!r}.")


LogReg = LogReg_Connectome
ElasticNet = ElasticNet_Connectome
SVM = SVM_Connectome


