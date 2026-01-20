import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import snntorch as snn
from snntorch import surrogate

# ==========================================
# 1. The Spiking Graph Encoder (SNN-GNN)
# ==========================================
class SpikingGNNEncoder(nn.Module):
    def __init__(self, num_node_features, hidden_dim, embedding_dim, beta=0.9):
        super().__init__()
        
        # Graph Convolution Layers (Spatial)
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embedding_dim)
        
        # Leaky Integrate-and-Fire Neurons (Temporal)
        # spike_grad is the surrogate gradient needed for backprop in SNNs
        spike_grad = surrogate.fast_sigmoid()
        
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x, edge_index, time_steps=10):
        """
        x: Node features [Num_Nodes, Features] (Static for simplicity, or time-series)
        edge_index: Graph connectivity
        time_steps: How long we simulate the brain (latency)
        """
        
        # Initialize membrane potential (voltage) for neurons
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        # Record the final spike output (or membrane potential) as the embedding
        spk2_rec = []
        
        # --- The Spiking Loop (Simulating Time) ---
        for step in range(time_steps):
            # 1. Spatial Aggregation (Graph Conv)
            cur1 = self.conv1(x, edge_index)
            
            # 2. Temporal Integration (LIF Neuron 1)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            # 3. Layer 2 (Input is the SPIKES from Layer 1)
            cur2 = self.conv2(spk1, edge_index) # Weighted by graph connections
            spk2, mem2 = self.lif2(cur2, mem2)
            
            # Record output
            spk2_rec.append(spk2)
            
        # Stack time steps: [Time, Num_Nodes, Embedding_Dim]
        output_spikes = torch.stack(spk2_rec, dim=0)
        
        # Aggregate over time to get a single "Concept Vector" (e.g., mean firing rate)
        # This is the "Joint Embedding"
        embedding = torch.mean(output_spikes, dim=0) 
        return embedding

# ==========================================
# 2. The Predictor (JEPA Logic)
# ==========================================
class JEPAPredictor(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        # Simple MLP to predict the Target Embedding from the Context Embedding
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, context_embedding):
        return self.net(context_embedding)

# ==========================================
# 3. The Full SG-JEPA Model
# ==========================================
class SpikingGraphJEPA(nn.Module):
    def __init__(self, num_features, hidden_dim, embedding_dim, beta=0.9, time_steps=10, mask_ratio=0.15):
        super().__init__()
        self.time_steps = time_steps
        self.mask_ratio = mask_ratio
        
        # --- Context Encoder (The Learner) ---
        self.context_encoder = SpikingGNNEncoder(num_features, hidden_dim, embedding_dim, beta=beta)
        
        # --- Target Encoder (The Teacher) ---
        # In JEPA, this is a copy of the Context Encoder, updated via EMA (not gradient descent)
        self.target_encoder = SpikingGNNEncoder(num_features, hidden_dim, embedding_dim, beta=beta)
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        
        # Prevent gradient updates for Target Encoder
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        # --- Predictor ---
        self.predictor = JEPAPredictor(embedding_dim, hidden_dim)

    def forward(self, x, edge_index, mask_indices=None, time_steps=None):
        """
        JEPA Training Step:
        1. Mask part of the input (Context).
        2. Keep the full input for the Target.
        3. Predict Target embedding from Context embedding.
        """
        if time_steps is None:
            time_steps = self.time_steps

        if mask_indices is None:
            num_nodes = x.size(0)
            num_mask = max(1, int(self.mask_ratio * num_nodes))
            mask_indices = torch.randperm(num_nodes, device=x.device)[:num_mask]
        else:
            mask_indices = mask_indices.to(x.device).long()
        
        # A. Get Target Embedding (Teacher)
        # The target sees the FULL graph (or the unmasked region)
        with torch.no_grad():
            target_embedding = self.target_encoder(x, edge_index, time_steps=time_steps)
        
        # B. Get Context Embedding (Student)
        # We simulate "Masking" by zeroing out features of specific nodes
        x_masked = x.clone()
        x_masked[mask_indices] = 0  # Simple masking strategy
        
        context_embedding = self.context_encoder(x_masked, edge_index, time_steps=time_steps)
        
        # C. Predict
        # The student tries to guess the Target's embedding for the masked nodes
        predicted_embedding = self.predictor(context_embedding)
        
        return predicted_embedding, target_embedding

    # --- EMA Update (Crucial for JEPA) ---
    def update_target_encoder(self, momentum=0.99):
        """
        Slowly updates the Target Encoder weights to match Context Encoder.
        This prevents the model from collapsing (outputting all zeros).
        """
        for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
