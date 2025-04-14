"""
This module serves as the central configuration hub for the Adaptive 
Multi-Intersection Traffic Signal Control project. It stores all 
hyperparameters, simulation settings, network definitions, file paths, 
and other constants required by various components of the system. 
Centralizing configuration simplifies parameter tuning, experiment management, 
and code maintenance by providing a single source of truth for settings.
"""

import os

# --- Simulation Settings ---
SUMO_EXECUTABLE = "sumo-gui" 
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SUMO_CONFIG_FILE = os.path.join(PROJECT_ROOT, "sumo_files", "scenario", "osm.sumocfg")
NET_FILE = os.path.join(PROJECT_ROOT, "sumo_files", "scenario", "osm.net.xml")
ROUTE_FILE = os.path.join(PROJECT_ROOT, "sumo_files", "scenario", "routes.rou.xml")

SIMULATION_STEP_LENGTH = 10.0 
DECISION_INTERVAL = 5 
EPISODE_LENGTH_SECONDS = 3600 

# --- Network Parameters ---
INTERSECTION_IDS = ['J2', 'J3', 'J6', 'J7'] 
NUM_INTERSECTIONS = len(INTERSECTION_IDS)
INTERSECTION_ID_TO_NODE_IDX = {id_: i for i, id_ in enumerate(INTERSECTION_IDS)}
NODE_IDX_TO_INTERSECTION_ID = {i: id_ for id_, i in INTERSECTION_ID_TO_NODE_IDX.items()}
ACTION_PHASE_MAP = {
    0: 0, # Action 0 -> Phase 0 (e.g., NS Green)
    1: 2  # Action 1 -> Phase 2 (e.g., EW Green)
}
NUM_ACTIONS_PER_INTERSECTION = len(ACTION_PHASE_MAP)
MIN_GREEN_TIME = 10
YELLOW_TIME = 3

# --- State Representation (Features) ---
MAX_QUEUE = 50.0
MAX_WAIT_TIME = 300.0
MAX_SPEED = 30.0 
MAX_DENSITY = 1.0 
MAX_TRAVEL_TIME = 100.0 
MAX_VEHICLE_COUNT = 50.0

NUM_NODE_FEATURES = 5 
NUM_EDGE_FEATURES = 4 
USE_EDGE_FEATURES = False 

# --- GNN Architecture ---
GNN_LAYER_TYPE = "GCNConv" 
GNN_HIDDEN_CHANNELS = 64
GNN_OUTPUT_CHANNELS = 64
GNN_ACTIVATION = "ReLU"

# --- DQN Hyperparameters (Stable Baselines3) ---
POLICY_TYPE = "MlpPolicy" 
LEARNING_RATE = 5e-4
BUFFER_SIZE = 100_000
LEARNING_STARTS = 1000
BATCH_SIZE = 64
GAMMA = 0.99 
TAU = 1.0 
TARGET_UPDATE_INTERVAL = 1000 
TRAIN_FREQ = 4
GRADIENT_STEPS = 1 
EXPLORATION_FRACTION = 0.3
EXPLORATION_INITIAL_EPS = 1.0
EXPLORATION_FINAL_EPS = 0.05

# --- Training Parameters ---
TOTAL_TRAINING_TIMESTEPS = 200_000
EVALUATION_FREQUENCY = 10_000
EVALUATION_EPISODES = 5
MODEL_SAVE_FREQUENCY = 20_000 

# --- Reward Function Weights ---
REWARD_QUEUE_WEIGHT = 0.3
REWARD_WAIT_TIME_WEIGHT = 0.7
GRIDLOCK_PENALTY = -500

# --- Logging and Saving Paths ---
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "models")
TENSORBOARD_LOG_NAME = "GNN_DQN_Traffic"

# --- Misc ---
RANDOM_SEED = 42