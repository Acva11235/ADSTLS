"""
This module defines the neural network architectures used by the RL agent, 
specifically the Graph Neural Network (GNN) based feature extractor and potentially 
the Q-network structure. The GNNFeatureExtractor processes the graph-based 
state representation (PyG Data object) received from the environment's 
observation space, extracting relevant spatial features and node embeddings. 
These features are then passed to the subsequent layers of the DQN's Q-network.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv 
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import numpy as np 

import config
import state_builder 


class GNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom Graph Neural Network feature extractor for Stable Baselines3.
    Processes observations structured as PyG Data objects (passed as Dict from env).
    Outputs a flattened feature vector representing the aggregated node embeddings.
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 0):
        # Calculate the output dimension from GNN
        gnn_output_dim_per_node = config.GNN_OUTPUT_CHANNELS
        self._features_dim = config.NUM_INTERSECTIONS * gnn_output_dim_per_node

        # Initialize BaseFeaturesExtractor AFTER calculating _features_dim
        super().__init__(observation_space, features_dim=self._features_dim)

        # Determine input channels based on node features
        input_node_channels = observation_space['x'].shape[1] # Get from space definition

        # Define GNN layers based on config
        hidden_channels = config.GNN_HIDDEN_CHANNELS
        output_channels = config.GNN_OUTPUT_CHANNELS
        activation_class = getattr(nn, config.GNN_ACTIVATION)

        # Layer 1
        if config.GNN_LAYER_TYPE == "GCNConv":
            self.conv1 = GCNConv(input_node_channels, hidden_channels)
        elif config.GNN_LAYER_TYPE == "GATConv":
            # Example: 8 attention heads
            self.conv1 = GATConv(input_node_channels, hidden_channels, heads=8, concat=True)
            hidden_channels = hidden_channels * 8 # Adjust hidden size due to concat
        elif config.GNN_LAYER_TYPE == "GATv2Conv":
            self.conv1 = GATv2Conv(input_node_channels, hidden_channels, heads=8, concat=True)
            hidden_channels = hidden_channels * 8 # Adjust hidden size due to concat
        else:
            raise ValueError(f"Unsupported GNN layer type: {config.GNN_LAYER_TYPE}")

        self.act1 = activation_class()

        # Layer 2 - Adjust input channels if using attention heads
        if config.GNN_LAYER_TYPE == "GCNConv":
            self.conv2 = GCNConv(hidden_channels, output_channels)
        elif "GAT" in config.GNN_LAYER_TYPE:
            # Output layer often averages heads instead of concatenating
            self.conv2 = GATConv(hidden_channels, output_channels, heads=1, concat=False)
            # self.conv2 = GATv2Conv(hidden_channels, output_channels, heads=1, concat=False) # If using GATv2
        else: # Should not happen if first layer check passed
            raise ValueError(f"Unsupported GNN layer type: {config.GNN_LAYER_TYPE}")

        self.act2 = activation_class()

        # Ensure static edge index is initialized in state_builder if needed
        state_builder._initialize_static_graph_structure()
        self.static_edge_index = state_builder._edge_index.to(self.device) # Move to device once


    def to(self, device):
        # Override `to` method to ensure static_edge_index is also moved
        super().to(device)
        if self.static_edge_index is not None:
            self.static_edge_index = self.static_edge_index.to(device)
        return self

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        # SB3 passes observations as a dictionary of batched tensors.
        # We need to reconstruct a PyG Batch object.

        x_batch = observations['x'] # Shape: [batch_size, num_nodes, num_features]
        batch_size = x_batch.shape[0]

        data_list = []
        for i in range(batch_size):
            x = x_batch[i] # [num_nodes, num_features]

            # Use the static edge_index, ensure it's on the correct device
            edge_index = self.static_edge_index.to(x.device)

            edge_attr = None
            if config.USE_EDGE_FEATURES and 'edge_attr' in observations:
                # Assuming edge_attr comes batched correctly
                # Shape needs to be [num_edges, num_edge_features] per item
                edge_attr_batch = observations['edge_attr'] # [batch_size, num_edges, num_features]
                if edge_attr_batch.shape[0] == batch_size:
                    edge_attr = edge_attr_batch[i] # [num_edges, num_features]
                else:
                    # Handle potential mismatch if SB3 batches differently
                    # This might happen if edge_attr is None for some items
                    # Defaulting to None if shapes don't align as expected
                    print(f"Warning: edge_attr batch size mismatch. Expected {batch_size}, Got {edge_attr_batch.shape[0]}")
                    edge_attr=None # Fallback


            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

        # Create PyG Batch object (handles node/edge offsets automatically)
        pyg_batch = Batch.from_data_list(data_list).to(x_batch.device)

        # Perform GNN forward pass
        h = self.conv1(pyg_batch.x, pyg_batch.edge_index, pyg_batch.edge_attr)
        h = self.act1(h)
        h = self.conv2(h, pyg_batch.edge_index) # edge_attr often optional for second layer
        node_embeddings = self.act2(h) # Shape: [batch_size * num_nodes, gnn_output_dim]

        # Reshape to get one feature vector per batch item
        # Flatten the node embeddings for each graph in the batch
        features = node_embeddings.view(batch_size, -1) # Shape: [batch_size, num_nodes * gnn_output_dim]

        # Ensure the output dimension matches the declared self._features_dim
        if features.shape[1] != self._features_dim:
            raise ValueError(f"Output feature dimension mismatch. Expected {self._features_dim}, Got {features.shape[1]}")

        return features


# Note: If a custom Q-network structure is needed beyond what `net_arch` allows,
# you would typically define a custom policy class inheriting from DQNPolicy
# and override `make_q_net` or potentially the entire `forward` method of the policy.
# For now, we rely on SB3's default handling combined with `net_arch` in policy_kwargs.