"""
This module is responsible for constructing the state representation for the 
RL agent. It queries raw simulation data using the `sumo_connector` module, 
processes this data into meaningful features (node and edge features), 
normalizes these features, and assembles them into a graph structure 
compatible with PyTorch Geometric (PyG). The output is typically a 
PyG `Data` object containing node features (`x`), edge connectivity (`edge_index`), 
and optionally edge features (`edge_attr`), representing the current traffic 
state of the controlled network.
"""

import torch
import numpy as np
from torch_geometric.data import Data
import sumolib

import sumo_connector
import config

_net = None
_edge_index = None
_edge_id_map = None # Maps edge_index columns to SUMO edge IDs if needed

def _initialize_static_graph_structure():
    global _net, _edge_index, _edge_id_map
    if _net is None:
        _net = sumolib.net.readNet(config.NET_FILE)

    if _edge_index is None:
        # Define node indices based on config
        node_indices = list(range(config.NUM_INTERSECTIONS)) # 0, 1, 2, 3

        # Manually define edges for a 2x2 grid, assuming specific node ordering
        # Example: 0=TopLeft, 1=TopRight, 2=BottomLeft, 3=BottomRight
        # Connections: (0,1), (1,0), (0,2), (2,0), (1,3), (3,1), (2,3), (3,2)
        # This needs to be adapted if intersection IDs or layout changes
        # A more robust approach would parse the SUMO net file to find edges
        # connecting the INTERSECTION_IDS.
        src_nodes = [0, 1, 0, 2, 1, 3, 2, 3]
        dst_nodes = [1, 0, 2, 0, 3, 1, 3, 2]

        # Placeholder for mapping edge_index columns to actual SUMO edge IDs
        # This mapping is CRUCIAL if edge features are used.
        # Example: You would need to know the SUMO edge ID connecting J2 to J3
        # _edge_id_map = { (0, 1): "edge_J2_J3", (1, 0): "edge_J3_J2", ... }

        _edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)

def _get_incoming_lanes(tls_id):
    # Cache results if network structure is static
    # This simplified version assumes sumo_connector provides a way or we parse the net
    # Let's try parsing the net directly here if not cached
    if _net is None:
        _initialize_static_graph_structure()

    incoming_lanes = []
    tls_node = _net.getNode(tls_id)
    for edge in tls_node.getIncoming():
        # Filter out edges coming from uncontrolled nodes if necessary
        # For now, assume all incoming edges are relevant for feature calculation
        for lane in edge.getLanes():
            incoming_lanes.append(lane.getID())
    return incoming_lanes


def _normalize(value, max_value):
    return min(value / (max_value + 1e-6), 1.0) # Add epsilon for stability


def build_state():
    global _net
    if _net is None or _edge_index is None:
        _initialize_static_graph_structure()

    num_nodes = config.NUM_INTERSECTIONS
    node_features_list = []
    edge_features_list = [] # Only populated if USE_EDGE_FEATURES is True

    # --- Calculate Node Features ---
    for node_idx in range(num_nodes):
        tls_id = config.NODE_IDX_TO_INTERSECTION_ID[node_idx]
        incoming_lanes = _get_incoming_lanes(tls_id)

        # 1. Phase Mask (One-Hot)
        current_phase_idx = sumo_connector.get_traffic_light_phase(tls_id)
        # Map SUMO phase index to simplified action/phase index (0 or 1)
        # This requires knowledge of the SUMO phase definitions
        # Example: If phase 0 is NS Green, phase 2 is EW Green
        phase_mask = [0.0] * config.NUM_ACTIONS_PER_INTERSECTION
        for action_idx, sumo_phase in config.ACTION_PHASE_MAP.items():
             if current_phase_idx == sumo_phase:
                 phase_mask[action_idx] = 1.0
                 break # Assuming one green phase active at a time for actions
        # Handle yellow/other phases if necessary (e.g., map to all zeros or previous green)
        # If no mapped phase found (e.g., yellow), maybe keep previous state or use zeros?
        # Using zeros if not explicitly mapped:
        if sum(phase_mask) == 0:
             # If it's yellow, we might want state to reflect previous green
             # This requires more complex state tracking in the environment
             # For now, zeros indicate non-actionable phase (like yellow)
             pass


        # 2. Elapsed Time in Phase (Normalized)
        # Note: SUMO's phase time info can be tricky. Using time since last switch.
        # Requires external tracking in the environment for accuracy during yellows.
        # sumo_connector.get_time_since_last_phase_switch(tls_id) is a placeholder
        # Using phase duration for now as a proxy, needs refinement in env.
        elapsed_time = sumo_connector.get_time_since_last_phase_switch(tls_id)
        # How to normalize? Max phase duration? Use decision interval?
        elapsed_norm = _normalize(elapsed_time, config.DECISION_INTERVAL * 5) # Norm guess

        # 3. Queue Sum (Normalized)
        total_halting = 0
        for lane_id in incoming_lanes:
            total_halting += sumo_connector.get_lane_halting_vehicle_count(lane_id)
        queue_sum_norm = _normalize(total_halting, config.MAX_QUEUE * len(incoming_lanes)) # Rough max

        # 4. Max Queue (Normalized)
        max_halting = 0
        for lane_id in incoming_lanes:
            max_halting = max(max_halting, sumo_connector.get_lane_halting_vehicle_count(lane_id))
        max_queue_norm = _normalize(max_halting, config.MAX_QUEUE)

        # 5. Total Wait Time (Normalized)
        total_wait_time = 0
        for lane_id in incoming_lanes:
            total_wait_time += sumo_connector.get_lane_waiting_time_sum(lane_id)
        # Normalization depends heavily on traffic volume and time scale
        wait_time_norm = _normalize(total_wait_time, config.MAX_WAIT_TIME * len(incoming_lanes)) # Rough max

        node_features = phase_mask + [elapsed_norm, queue_sum_norm, max_queue_norm, wait_time_norm]
        # Verify feature count matches config.NUM_NODE_FEATURES
        if len(node_features) != config.NUM_NODE_FEATURES:
             print(f"Warning: Feature count mismatch. Expected {config.NUM_NODE_FEATURES}, got {len(node_features)}")
             # Adjust padding or fix calculation
             # Example padding:
             node_features.extend([0.0] * (config.NUM_NODE_FEATURES - len(node_features)))
             node_features = node_features[:config.NUM_NODE_FEATURES]


        node_features_list.append(node_features)

    x = torch.tensor(node_features_list, dtype=torch.float)

    # --- Calculate Edge Features (Optional) ---
    edge_attr = None
    if config.USE_EDGE_FEATURES:
        num_edges = _edge_index.shape[1]
        for i in range(num_edges):
            src_node_idx = _edge_index[0, i].item()
            dst_node_idx = _edge_index[1, i].item()

            # Need mapping from (src_node_idx, dst_node_idx) to SUMO edge ID(s)
            # sumo_edge_id = _edge_id_map.get((src_node_idx, dst_node_idx))
            sumo_edge_id = None # Placeholder - Requires actual mapping logic

            if sumo_edge_id:
                mean_speed = sumo_connector.get_edge_mean_speed(sumo_edge_id)
                density = sumo_connector.get_edge_density(sumo_edge_id)
                travel_time = sumo_connector.get_edge_travel_time(sumo_edge_id)
                veh_count = sumo_connector.get_edge_vehicle_count(sumo_edge_id)

                mean_speed_norm = _normalize(mean_speed, config.MAX_SPEED)
                density_norm = _normalize(density, config.MAX_DENSITY)
                travel_time_norm = _normalize(travel_time, config.MAX_TRAVEL_TIME)
                veh_count_norm = _normalize(veh_count, config.MAX_VEHICLE_COUNT)

                edge_features = [mean_speed_norm, density_norm, travel_time_norm, veh_count_norm]
            else:
                # Default features if edge ID mapping fails
                edge_features = [0.0] * config.NUM_EDGE_FEATURES

            edge_features_list.append(edge_features)

        if edge_features_list:
             # Verify feature count
             if len(edge_features_list[0]) != config.NUM_EDGE_FEATURES:
                  print(f"Warning: Edge Feature count mismatch. Expected {config.NUM_EDGE_FEATURES}, got {len(edge_features_list[0])}")
                  # Adjust or fix
                  for i in range(len(edge_features_list)):
                       edge_features_list[i].extend([0.0] * (config.NUM_EDGE_FEATURES - len(edge_features_list[i])))
                       edge_features_list[i] = edge_features_list[i][:config.NUM_EDGE_FEATURES]

             edge_attr = torch.tensor(edge_features_list, dtype=torch.float)


    # --- Assemble PyG Data Object ---
    state_data = Data(x=x, edge_index=_edge_index, edge_attr=edge_attr)

    return state_data