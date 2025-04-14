"""
This module is responsible for setting up and managing the Reinforcement 
Learning agent, specifically the Deep Q-Networks (DQN) agent from the 
Stable Baselines3 library. It configures the agent with the custom 
traffic environment, the GNN feature extractor, hyperparameters loaded 
from the config module, and policy network architecture details. It provides 
functions for creating, saving, and loading the DQN agent.
"""

import os
import torch
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

import config
from traffic_env import TrafficEnv # Import the custom environment
from network import GNNFeatureExtractor # Import the custom feature extractor

def setup_agent(env, load_checkpoint_path=None):
    """
    Sets up and returns a Stable Baselines3 DQN agent.

    Args:
        env: The instantiated TrafficEnv environment instance.
        load_checkpoint_path (str, optional): Path to a saved model checkpoint to load. Defaults to None.

    Returns:
        stable_baselines3.DQN: The configured DQN agent.
    """
    # Set seeds for reproducibility
    if config.RANDOM_SEED is not None:
        torch.manual_seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)
        # Note: Environment seeding is handled in env.reset() called by SB3

    # Define policy keyword arguments, including the custom feature extractor
    policy_kwargs = dict(
        features_extractor_class=GNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=config.NUM_INTERSECTIONS * config.GNN_OUTPUT_CHANNELS), # Match extractor's output dim
        # Define network layers AFTER the feature extractor
        # Example: Two hidden layers of size 128
        # SB3 DQN's MlpPolicy will construct Q-network head based on action space
        net_arch=[128, 128] # Adjust layer sizes as needed
    )

    # Check if loading an existing model
    if load_checkpoint_path and os.path.exists(load_checkpoint_path):
        print(f"Loading existing model from: {load_checkpoint_path}")
        # When loading, SB3 automatically uses the saved policy_kwargs and parameters,
        # but we need to provide the environment.
        # Custom objects (like GNNFeatureExtractor) need to be available in the scope.
        agent = DQN.load(
            load_checkpoint_path,
            env=env,
            custom_objects={'policy_kwargs': policy_kwargs}, # May be needed if SB3 doesn't auto-find extractor
            tensorboard_log=config.LOG_DIR,
            seed=config.RANDOM_SEED,
            device='auto' # Or specify 'cuda', 'cpu'
        )
        print("Model loaded successfully.")
    else:
        if load_checkpoint_path:
            print(f"Warning: Checkpoint path specified but not found: {load_checkpoint_path}. Creating new agent.")

        print("Creating new DQN agent...")
        # Instantiate the DQN agent
        agent = DQN(
            policy=config.POLICY_TYPE, 
            env=env,
            learning_rate=config.LEARNING_RATE,
            buffer_size=config.BUFFER_SIZE,
            learning_starts=config.LEARNING_STARTS,
            batch_size=config.BATCH_SIZE,
            tau=config.TAU,
            gamma=config.GAMMA,
            train_freq=config.TRAIN_FREQ, 
            gradient_steps=config.GRADIENT_STEPS,
            target_update_interval=config.TARGET_UPDATE_INTERVAL,
            exploration_fraction=config.EXPLORATION_FRACTION,
            exploration_initial_eps=config.EXPLORATION_INITIAL_EPS,
            exploration_final_eps=config.EXPLORATION_FINAL_EPS,
            policy_kwargs=policy_kwargs,
            tensorboard_log=config.LOG_DIR,
            seed=config.RANDOM_SEED,
            verbose=1, # 0 = no output, 1 = info, 2 = debug
            device='auto' # Automatically use GPU if available
        )
        print("New agent created.")

    return agent


def save_agent(agent, save_path):
    """
    Saves the agent's model to the specified path.

    Args:
        agent (stable_baselines3.DQN): The agent to save.
        save_path (str): The file path to save the model (e.g., 'model.zip').
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        agent.save(save_path)
        print(f"Agent saved successfully to: {save_path}")
    except Exception as e:
        print(f"Error saving agent to {save_path}: {e}")


def load_agent(load_path, env):
    """
    Loads a pre-trained agent model.

    Args:
        load_path (str): The file path of the saved model.
        env: The environment instance (required by SB3 load).

    Returns:
        stable_baselines3.DQN: The loaded agent.
    """
    if not os.path.exists(load_path):
        print(f"Error: Model file not found at {load_path}")
        return None

    try:
        # Need to provide custom_objects if they are not automatically found
        # policy_kwargs might be needed if the feature extractor isn't standard
        policy_kwargs = dict(
            features_extractor_class=GNNFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=config.NUM_INTERSECTIONS * config.GNN_OUTPUT_CHANNELS),
            net_arch=[128, 128] # Should match the saved model structure
        )
        agent = DQN.load(
            load_path,
            env=env,
            custom_objects={'policy_kwargs': policy_kwargs}, # Pass custom objects
            device='auto'
        )
        print(f"Agent loaded successfully from: {load_path}")
        return agent
    except Exception as e:
        print(f"Error loading agent from {load_path}: {e}")
        return None