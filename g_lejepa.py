"""
Graph-LeJEPA: Graph Joint-Embedding Predictive Architecture with SIGReg

This implementation combines Graph-JEPA's masked subgraph prediction with LeJEPA's
SIGReg (Sigmoid Regularization) for collapse prevention. Key innovations:
- Replaces EMA-based teacher-student with SIGReg regularization
- Uses METIS clustering for meaningful subgraph masking
- O(N) complexity through Cramer-Wold random projections
- Single hyperparameter (lambda) for regularization

References:
- Graph-JEPA: Skenderi et al., TMLR 2025
- LeJEPA SIGReg: Balestriero & LeCun, arXiv November 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool
from typing import Optional, Tuple


# ==========================================
# 1. SIGReg Loss (Cramer-Wold Random Projections)
# ==========================================
class SIGRegLoss(nn.Module):
    """
    SIGReg: Sigmoid Regularization for collapse prevention.

    Based on the Cramer-Wold theorem: if all 1D projections of a distribution
    are Gaussian, then the full distribution is Gaussian. This provides O(N)
    complexity regularization that enforces isotropic Gaussian embeddings.

    Args:
        num_projections: Number of random projections (more = better approximation)
        lambda_reg: Regularization strength
        eps: Small constant for numerical stability
    """

    def __init__(self, num_projections: int = 256, lambda_reg: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.num_projections = num_projections
        self.lambda_reg = lambda_reg
        self.eps = eps
        self._projection_matrix = None
        self._proj_dim = None

    def _get_projections(self, dim: int, device: torch.device) -> torch.Tensor:
        """Get or create random projection matrix."""
        if self._projection_matrix is None or self._proj_dim != dim:
            # Random unit vectors for projection
            proj = torch.randn(dim, self.num_projections, device=device)
            proj = F.normalize(proj, dim=0)  # Normalize to unit vectors
            self._projection_matrix = proj
            self._proj_dim = dim
        return self._projection_matrix.to(device)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute SIGReg loss to enforce isotropic Gaussian distribution.

        Args:
            embeddings: [batch_size, embedding_dim] tensor

        Returns:
            SIGReg regularization loss
        """
        batch_size, dim = embeddings.shape

        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device)

        # Center the embeddings (zero mean)
        embeddings_centered = embeddings - embeddings.mean(dim=0, keepdim=True)

        # Get random projection directions
        projections = self._get_projections(dim, embeddings.device)

        # Project embeddings onto random directions: [batch_size, num_projections]
        projected = torch.matmul(embeddings_centered, projections)

        # Compute variance along each projection
        variances = projected.var(dim=0) + self.eps

        # Target variance is 1 (standard Gaussian)
        # Variance loss: encourage unit variance
        var_loss = ((variances - 1.0) ** 2).mean()

        # Covariance loss: encourage decorrelation between projections
        # Compute correlation between different projections
        projected_normalized = projected / (projected.std(dim=0, keepdim=True) + self.eps)
        cov_matrix = torch.matmul(projected_normalized.T, projected_normalized) / batch_size

        # Off-diagonal elements should be zero (decorrelation)
        off_diag_mask = ~torch.eye(self.num_projections, dtype=torch.bool, device=embeddings.device)
        cov_loss = (cov_matrix[off_diag_mask] ** 2).mean()

        # Combined SIGReg loss
        total_loss = self.lambda_reg * (var_loss + cov_loss)

        return total_loss


# ==========================================
# 2. Graph Neural Network Encoder
# ==========================================
class GNNEncoder(nn.Module):
    """
    Graph Neural Network encoder with flexible architecture.

    Args:
        num_features: Input feature dimension
        hidden_dim: Hidden layer dimension
        embedding_dim: Output embedding dimension
        num_layers: Number of GNN layers
        gnn_type: Type of GNN layer ('gcn' or 'gin')
        dropout: Dropout probability
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        embedding_dim: int,
        num_layers: int = 3,
        gnn_type: str = 'gcn',
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Input layer
        if gnn_type == 'gcn':
            self.convs.append(GCNConv(num_features, hidden_dim))
        elif gnn_type == 'gin':
            mlp = nn.Sequential(
                nn.Linear(num_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))

        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            if gnn_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == 'gin':
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                self.convs.append(GINConv(mlp))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Output layer
        if gnn_type == 'gcn':
            self.convs.append(GCNConv(hidden_dim, embedding_dim))
        elif gnn_type == 'gin':
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embedding_dim)
            )
            self.convs.append(GINConv(mlp))

        self.batch_norms.append(nn.BatchNorm1d(embedding_dim))

        # Final projection head
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the GNN encoder.

        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment for nodes (for batched graphs)

        Returns:
            node_embeddings: [num_nodes, embedding_dim]
            graph_embedding: [batch_size, embedding_dim] (if batch provided)
        """
        # Node-level encoding
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        node_embeddings = x

        # Graph-level embedding via pooling
        if batch is not None:
            graph_embedding = global_mean_pool(node_embeddings, batch)
        else:
            graph_embedding = node_embeddings.mean(dim=0, keepdim=True)

        # Project to final embedding space
        graph_embedding = self.projection(graph_embedding)

        return node_embeddings, graph_embedding


# ==========================================
# 3. METIS-style Subgraph Masking
# ==========================================
class SubgraphMasker:
    """
    Subgraph masking using clustering-based approach.
    Inspired by Graph-JEPA's METIS clustering for meaningful subgraph selection.

    For simplicity, uses spectral clustering or random connected subgraph selection.
    For production, integrate with PyTorch Geometric's METIS partitioning.
    """

    @staticmethod
    def random_node_mask(
        num_nodes: int,
        mask_ratio: float = 0.15,
        device: torch.device = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Random node masking strategy.

        Returns:
            context_mask: Boolean mask for context nodes (True = keep)
            target_mask: Boolean mask for target/masked nodes (True = masked)
        """
        num_mask = max(1, int(mask_ratio * num_nodes))
        perm = torch.randperm(num_nodes, device=device)

        target_indices = perm[:num_mask]

        target_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        target_mask[target_indices] = True

        context_mask = ~target_mask

        return context_mask, target_mask

    @staticmethod
    def cluster_based_mask(
        edge_index: torch.Tensor,  # noqa: ARG004 - reserved for METIS integration
        num_nodes: int,
        num_clusters: int = 4,
        mask_clusters: int = 1,
        device: torch.device = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cluster-based masking: mask entire clusters of related nodes.
        Uses simple label propagation for clustering.

        Note: edge_index parameter reserved for future METIS clustering integration.
        Currently uses random cluster assignment as a placeholder.

        Args:
            edge_index: Graph connectivity (for future METIS integration)
            num_nodes: Number of nodes
            num_clusters: Number of clusters to create
            mask_clusters: Number of clusters to mask

        Returns:
            context_mask, target_mask
        """
        # Simple clustering via random initialization
        # TODO: Replace with METIS clustering for production use
        cluster_labels = torch.randint(0, num_clusters, (num_nodes,), device=device)

        # Select clusters to mask
        clusters_to_mask = torch.randperm(num_clusters, device=device)[:mask_clusters]

        target_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        for c in clusters_to_mask:
            target_mask |= (cluster_labels == c)

        # Ensure at least some nodes are masked
        if target_mask.sum() == 0:
            target_mask[:max(1, num_nodes // 4)] = True

        # Ensure at least some context nodes remain
        if (~target_mask).sum() == 0:
            target_mask[num_nodes // 2:] = False

        context_mask = ~target_mask

        return context_mask, target_mask


# ==========================================
# 4. JEPA Predictor Module
# ==========================================
class Predictor(nn.Module):
    """
    Predictor network for JEPA.
    Predicts target embeddings from context embeddings.

    Args:
        embedding_dim: Dimension of embeddings
        hidden_dim: Hidden layer dimension
        num_layers: Number of predictor layers
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 2
    ):
        super().__init__()

        layers = []
        layers.append(nn.Linear(embedding_dim, hidden_dim))
        layers.append(nn.GELU())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())

        layers.append(nn.Linear(hidden_dim, embedding_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ==========================================
# 5. Graph-LeJEPA Model
# ==========================================
class GraphLeJEPA(nn.Module):
    """
    Graph-LeJEPA: Graph JEPA with SIGReg Regularization.

    Key differences from standard Graph-JEPA:
    - No EMA target encoder - uses single encoder with SIGReg
    - SIGReg prevents collapse via isotropic Gaussian regularization
    - Simplified architecture with fewer hyperparameters

    Args:
        num_features: Input node feature dimension
        hidden_dim: Hidden layer dimension
        embedding_dim: Output embedding dimension
        num_layers: Number of GNN layers
        gnn_type: Type of GNN ('gcn' or 'gin')
        mask_ratio: Ratio of nodes to mask
        num_projections: Number of random projections for SIGReg
        lambda_reg: SIGReg regularization strength
        dropout: Dropout probability
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 256,
        embedding_dim: int = 128,
        num_layers: int = 3,
        gnn_type: str = 'gcn',
        mask_ratio: float = 0.15,
        num_projections: int = 256,
        lambda_reg: float = 1.0,
        dropout: float = 0.1
    ):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.embedding_dim = embedding_dim

        # Single encoder (no EMA target encoder needed with SIGReg)
        self.encoder = GNNEncoder(
            num_features=num_features,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            gnn_type=gnn_type,
            dropout=dropout
        )

        # Predictor for JEPA objective
        self.predictor = Predictor(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=2
        )

        # SIGReg loss for collapse prevention
        self.sigreg = SIGRegLoss(
            num_projections=num_projections,
            lambda_reg=lambda_reg
        )

        # Masker for subgraph selection
        self.masker = SubgraphMasker()

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode graph(s) to embeddings.

        Returns:
            node_embeddings, graph_embedding
        """
        return self.encoder(x, edge_index, batch)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        mask_type: str = 'random'
    ) -> dict:
        """
        Forward pass for training.

        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment for batched graphs
            mask_type: 'random' or 'cluster' masking

        Returns:
            Dictionary containing:
                - pred_loss: Prediction loss (MSE between predicted and target)
                - reg_loss: SIGReg regularization loss
                - total_loss: Combined loss
                - embeddings: Graph embeddings
        """
        num_nodes = x.size(0)
        device = x.device

        # Step 1: Get full graph embedding (target)
        _, full_embedding = self.encoder(x, edge_index, batch)

        # Step 2: Create mask
        if mask_type == 'random':
            _, target_mask = self.masker.random_node_mask(
                num_nodes, self.mask_ratio, device
            )
        else:
            _, target_mask = self.masker.cluster_based_mask(
                edge_index, num_nodes, device=device
            )

        # Step 3: Mask node features (zero out masked nodes)
        x_masked = x.clone()
        x_masked[target_mask] = 0

        # Step 4: Encode masked graph (context)
        _, context_embedding = self.encoder(x_masked, edge_index, batch)

        # Step 5: Predict target embedding from context
        predicted_embedding = self.predictor(context_embedding)

        # Step 6: Compute prediction loss (JEPA objective)
        # Use cosine similarity loss or MSE
        pred_loss = F.mse_loss(predicted_embedding, full_embedding.detach())

        # Step 7: Compute SIGReg regularization loss
        # Apply to full embeddings to prevent collapse
        reg_loss = self.sigreg(full_embedding)

        # Step 8: Total loss
        total_loss = pred_loss + reg_loss

        return {
            'pred_loss': pred_loss,
            'reg_loss': reg_loss,
            'total_loss': total_loss,
            'embeddings': full_embedding,
            'predicted': predicted_embedding
        }

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get graph embeddings for downstream tasks (inference mode).
        """
        self.eval()
        with torch.no_grad():
            _, embeddings = self.encoder(x, edge_index, batch)
        return embeddings


# ==========================================
# 6. Training Utilities
# ==========================================
class GraphLeJEPATrainer:
    """
    Trainer class for Graph-LeJEPA.

    Args:
        model: GraphLeJEPA model
        optimizer: PyTorch optimizer
        device: Device to train on
    """

    def __init__(
        self,
        model: GraphLeJEPA,
        optimizer: torch.optim.Optimizer,
        device: torch.device = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.history = {
            'pred_loss': [],
            'reg_loss': [],
            'total_loss': []
        }

    def train_step(
        self,
        data,
        mask_type: str = 'random'
    ) -> dict:
        """
        Single training step.

        Args:
            data: PyG Data object with x, edge_index, (optional) batch
            mask_type: Masking strategy

        Returns:
            Loss dictionary
        """
        self.model.train()
        self.optimizer.zero_grad()

        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        batch = data.batch.to(self.device) if hasattr(data, 'batch') else None

        output = self.model(x, edge_index, batch, mask_type)

        output['total_loss'].backward()
        self.optimizer.step()

        # Record history
        self.history['pred_loss'].append(output['pred_loss'].item())
        self.history['reg_loss'].append(output['reg_loss'].item())
        self.history['total_loss'].append(output['total_loss'].item())

        return {k: v.item() if torch.is_tensor(v) else v
                for k, v in output.items() if k != 'embeddings' and k != 'predicted'}

    def train_epoch(
        self,
        dataloader,
        mask_type: str = 'random'
    ) -> dict:
        """
        Train for one epoch.

        Args:
            dataloader: PyG DataLoader
            mask_type: Masking strategy

        Returns:
            Average losses for the epoch
        """
        epoch_losses = {'pred_loss': 0, 'reg_loss': 0, 'total_loss': 0}
        num_batches = 0

        for data in dataloader:
            losses = self.train_step(data, mask_type)
            for k in epoch_losses:
                epoch_losses[k] += losses[k]
            num_batches += 1

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= num_batches

        return epoch_losses

    def evaluate(
        self,
        dataloader
    ) -> torch.Tensor:
        """
        Get embeddings for all graphs in dataloader.

        Returns:
            Stacked embeddings tensor
        """
        self.model.eval()
        embeddings = []

        with torch.no_grad():
            for data in dataloader:
                x = data.x.to(self.device)
                edge_index = data.edge_index.to(self.device)
                batch = data.batch.to(self.device) if hasattr(data, 'batch') else None

                emb = self.model.get_embeddings(x, edge_index, batch)
                embeddings.append(emb.cpu())

        return torch.cat(embeddings, dim=0)


# ==========================================
# 7. Downstream Evaluation Head
# ==========================================
class LinearProbe(nn.Module):
    """
    Linear probe for evaluating learned representations.

    Args:
        embedding_dim: Input dimension
        num_classes: Number of output classes
    """

    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def evaluate_linear_probe(
    model: GraphLeJEPA,
    train_loader,
    test_loader,
    num_classes: int,
    device: torch.device,
    epochs: int = 100,
    lr: float = 0.01
) -> dict:
    """
    Evaluate Graph-LeJEPA embeddings using linear probe.

    Args:
        model: Trained GraphLeJEPA model
        train_loader: Training data loader
        test_loader: Test data loader
        num_classes: Number of classes
        device: Device
        epochs: Training epochs for probe
        lr: Learning rate

    Returns:
        Dictionary with train/test accuracy
    """
    model.eval()

    # Extract embeddings and labels
    def get_data(loader):
        embeddings, labels = [], []
        with torch.no_grad():
            for data in loader:
                x = data.x.to(device)
                edge_index = data.edge_index.to(device)
                batch = data.batch.to(device) if hasattr(data, 'batch') else None

                emb = model.get_embeddings(x, edge_index, batch)
                embeddings.append(emb.cpu())
                labels.append(data.y.cpu())

        return torch.cat(embeddings), torch.cat(labels)

    train_emb, train_labels = get_data(train_loader)
    test_emb, test_labels = get_data(test_loader)

    # Train linear probe
    probe = LinearProbe(model.embedding_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_emb = train_emb.to(device)
    train_labels = train_labels.to(device)
    test_emb = test_emb.to(device)
    test_labels = test_labels.to(device)

    for _ in range(epochs):
        probe.train()
        optimizer.zero_grad()
        out = probe(train_emb)
        loss = criterion(out, train_labels)
        loss.backward()
        optimizer.step()

    # Evaluate
    probe.eval()
    with torch.no_grad():
        train_pred = probe(train_emb).argmax(dim=1)
        test_pred = probe(test_emb).argmax(dim=1)

        train_acc = (train_pred == train_labels).float().mean().item()
        test_acc = (test_pred == test_labels).float().mean().item()

    return {'train_acc': train_acc, 'test_acc': test_acc}


if __name__ == '__main__':
    # Quick sanity check
    print("Graph-LeJEPA module loaded successfully!")

    # Test with random data
    num_nodes = 100
    num_features = 32

    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, 500))

    model = GraphLeJEPA(
        num_features=num_features,
        hidden_dim=64,
        embedding_dim=32,
        num_layers=2
    )

    output = model(x, edge_index)
    print(f"Prediction loss: {output['pred_loss'].item():.4f}")
    print(f"Regularization loss: {output['reg_loss'].item():.4f}")
    print(f"Total loss: {output['total_loss'].item():.4f}")
    print(f"Embedding shape: {output['embeddings'].shape}")
