"""
This module defines the reinforcement learning environment for the traffic signal
control problem, adhering to the Gymnasium (formerly OpenAI Gym) API. It uses 
the `sumo_connector` to interact with the SUMO simulation and the 
`state_builder` to generate the graph-based state representation. The environment
handles agent actions (traffic light phase changes), simulation stepping, 
reward calculation based on traffic metrics (queue lengths, waiting times), 
and episode termination conditions. It manages internal state like minimum 
green times and yellow phase transitions.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci # Needed for exception handling maybe

import sumo_connector
import state_builder
import config

class TrafficEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, use_gui=False, render_mode=None):
        super().__init__()

        self.use_gui = use_gui
        self.render_mode = render_mode # For potential future rendering logic

        # Define action space: MultiDiscrete (one discrete choice per intersection)
        self.action_space = spaces.MultiDiscrete([config.NUM_ACTIONS_PER_INTERSECTION] * config.NUM_INTERSECTIONS)

        # Define observation space:
        # This is complex because the actual observation is a PyG Data object.
        # SB3's custom feature extractor handles this conversion.
        # For Gym compliance, we define a placeholder space. A Box space
        # large enough to potentially hold flattened data is one option,
        # or a Dict space outlining the components. Let's use Dict.
        obs_space_dict = {
            'x': spaces.Box(low=0.0, high=1.0, shape=(config.NUM_INTERSECTIONS, config.NUM_NODE_FEATURES), dtype=np.float32),
            'edge_index': spaces.Box(low=0, high=config.NUM_INTERSECTIONS - 1, shape=(2, 8), dtype=np.int64) # Assuming 8 edges for 2x2 grid
            # Add edge_attr if used
        }
        if config.USE_EDGE_FEATURES:
             # Assuming edge index shape is known/fixed
             num_edges = state_builder._edge_index.shape[1] if state_builder._edge_index is not None else 8 # Default guess
             obs_space_dict['edge_attr'] = spaces.Box(low=0.0, high=1.0, shape=(num_edges, config.NUM_EDGE_FEATURES), dtype=np.float32)

        self.observation_space = spaces.Dict(obs_space_dict)


        self._simulation_step_count = 0
        self._episode_step_count = 0

        # Internal state tracking for minimum green times and current phases
        self._current_sumo_phase = [-1] * config.NUM_INTERSECTIONS
        self._time_since_last_action = [0] * config.NUM_INTERSECTIONS
        self._last_action_was_green = [False] * config.NUM_INTERSECTIONS


    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Handles seeding for reproducibility

        sumo_connector.close_simulation()
        sumo_connector.start_simulation(use_gui=self.use_gui)

        self._simulation_step_count = 0
        self._episode_step_count = 0
        self._current_sumo_phase = [-1] * config.NUM_INTERSECTIONS
        self._time_since_last_action = [0] * config.NUM_INTERSECTIONS
        self._last_action_was_green = [False] * config.NUM_INTERSECTIONS


        # Run a few steps to stabilize simulation if needed
        for _ in range(5):
             sumo_connector.simulation_step()
             self._simulation_step_count += config.SIMULATION_STEP_LENGTH

        observation = self._get_observation()
        info = self._get_info() # Return initial info

        # Update internal phase tracking after getting initial observation
        for i in range(config.NUM_INTERSECTIONS):
             tls_id = config.NODE_IDX_TO_INTERSECTION_ID[i]
             self._current_sumo_phase[i] = sumo_connector.get_traffic_light_phase(tls_id)
             # Check if initial phase is a green phase defined in ACTION_PHASE_MAP
             self._last_action_was_green[i] = self._current_sumo_phase[i] in config.ACTION_PHASE_MAP.values()


        return observation, info

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        current_rewards = np.zeros(config.NUM_INTERSECTIONS)
        applied_action_flags = [False] * config.NUM_INTERSECTIONS

        # --- Apply Actions with Yellow Phase and Min Green Time ---
        for i in range(config.NUM_INTERSECTIONS):
            tls_id = config.NODE_IDX_TO_INTERSECTION_ID[i]
            agent_action_idx = action[i] # The index chosen by the agent (e.g., 0 or 1)
            target_sumo_phase_idx = config.ACTION_PHASE_MAP[agent_action_idx]

            current_sumo_phase = sumo_connector.get_traffic_light_phase(tls_id)
            self._current_sumo_phase[i] = current_sumo_phase # Update internal record

            time_in_current_phase = self._time_since_last_action[i]
            is_currently_green = self._last_action_was_green[i]


            # Check minimum green time constraint ONLY if currently in a green phase
            if is_currently_green and time_in_current_phase < config.MIN_GREEN_TIME:
                 # Minimum green time not met, continue current phase
                 # Do not apply agent's action for this intersection yet
                 continue # Move to next intersection

            # Check if action requests a change from the current target green phase
            needs_change = False
            if is_currently_green:
                 # Agent wants to switch FROM a green phase TO a potentially different one
                 if target_sumo_phase_idx != current_sumo_phase:
                      needs_change = True
            else:
                 # Currently in yellow/red, agent wants to transition TO a green phase
                 # Allow transition regardless of target_sumo_phase_idx vs current_sumo_phase
                 needs_change = True


            if needs_change:
                 applied_action_flags[i] = True # Mark that we are changing this light

                 # If currently green, insert yellow phase first
                 if is_currently_green:
                     # Find the yellow phase corresponding to the current green phase
                     # This requires knowledge of SUMO phase definitions (e.g., phase N+1 is yellow for phase N)
                     yellow_phase_idx = current_sumo_phase + 1 # COMMON convention, adjust if needed
                     sumo_connector.set_traffic_light_phase(tls_id, yellow_phase_idx)
                     # Let yellow phase run for YELLOW_TIME seconds
                     for _ in range(int(config.YELLOW_TIME / config.SIMULATION_STEP_LENGTH)):
                         if not sumo_connector.simulation_step(): return self._handle_sim_error()
                         self._simulation_step_count += config.SIMULATION_STEP_LENGTH
                         # Intermediate reward calculation could happen here if needed

                     # Now set the target green phase
                     sumo_connector.set_traffic_light_phase(tls_id, target_sumo_phase_idx)
                     self._current_sumo_phase[i] = target_sumo_phase_idx # Update internal record
                     self._time_since_last_action[i] = 0 # Reset timer for new green phase
                     self._last_action_was_green[i] = True # Mark as green phase active

                 else: # Currently yellow/red, directly set target green phase
                     sumo_connector.set_traffic_light_phase(tls_id, target_sumo_phase_idx)
                     self._current_sumo_phase[i] = target_sumo_phase_idx
                     self._time_since_last_action[i] = 0
                     self._last_action_was_green[i] = True

        # --- Simulate Remaining Time in Decision Interval ---
        start_step = self._simulation_step_count
        # Adjust remaining steps based on whether yellow phases were run
        # This simplified loop assumes yellow phases fit within one master loop iteration
        # A more precise implementation might integrate yellow steps better.
        steps_to_run = int(config.DECISION_INTERVAL / config.SIMULATION_STEP_LENGTH)

        for step in range(steps_to_run):
            sim_ok = sumo_connector.simulation_step()
            if not sim_ok: return self._handle_sim_error()
            self._simulation_step_count += config.SIMULATION_STEP_LENGTH

            # Increment timers for intersections that didn't just change phase
            for i in range(config.NUM_INTERSECTIONS):
                if not applied_action_flags[i]:
                     self._time_since_last_action[i] += config.SIMULATION_STEP_LENGTH
                # Check if the current phase is still the expected green phase
                # SUMO might auto-transition if max duration is hit
                tls_id = config.NODE_IDX_TO_INTERSECTION_ID[i]
                live_phase = sumo_connector.get_traffic_light_phase(tls_id)
                if live_phase != self._current_sumo_phase[i]:
                     # Phase changed unexpectedly (e.g., max time, or it's now yellow)
                     self._current_sumo_phase[i] = live_phase
                     self._time_since_last_action[i] = config.SIMULATION_STEP_LENGTH # Just started this phase
                     self._last_action_was_green[i] = live_phase in config.ACTION_PHASE_MAP.values()
                elif self._last_action_was_green[i]:
                     # If phase didn't change AND it was green, increment timer
                     # This line seems redundant with the increment in applied_action_flags[i] loop?
                     # Let's ensure timers only increment once per step correctly
                     pass # Timer was incremented above if flag was false

            # Re-evaluate timers for those lights that DID change this step
            # Their timer starts AFTER the yellow phase (if any)
            for i in range(config.NUM_INTERSECTIONS):
                if applied_action_flags[i]:
                    # Correct timer increment after potential yellow phase
                    # If yellow = 3s, step = 1s, interval = 5s
                    # Yellow runs for 3 steps. Green runs for 2 steps.
                    # Timer should be 2s at the end of interval.
                    # This logic is getting complex, simplifying for now:
                    # Assume _time_since_last_action[i] correctly tracks time in *current* phase
                    current_phase_duration = sumo_connector.get_time_since_last_phase_switch(tls_id) # Approximation
                    self._time_since_last_action[i] = current_phase_duration


            # Optional: Accumulate rewards per step if needed
            # current_rewards += self._calculate_step_reward() / steps_to_run

        # --- Calculate Final Reward for the Interval ---
        reward = self._calculate_interval_reward()

        # --- Get Next Observation ---
        observation = self._get_observation()

        # --- Check Termination ---
        self._episode_step_count += 1
        terminated = (self._simulation_step_count >= config.EPISODE_LENGTH_SECONDS)
        truncated = False # Add truncation conditions if needed (e.g., gridlock detection)

        # Check for gridlock (simple version: very high total wait time)
        # total_wait = sum(self._get_all_waiting_times())
        # if total_wait > config.GRIDLOCK_THRESHOLD: # Need to define threshold
        #      truncated = True
        #      reward += config.GRIDLOCK_PENALTY


        info = self._get_info()
        info['step_reward'] = reward # Add reward to info

        if self.render_mode == "human":
            self.render() # Placeholder for rendering logic

        return observation, reward, terminated, truncated, info


    def _get_observation(self):
        # Use state_builder to get the PyG Data object
        pyg_data = state_builder.build_state()

        # For Gym compatibility with Dict space, convert PyG data to dict of arrays/tensors
        # The custom SB3 feature extractor will need to reconstruct the PyG object
        obs_dict = {
            'x': pyg_data.x.numpy(),
            'edge_index': pyg_data.edge_index.numpy()
        }
        if config.USE_EDGE_FEATURES and pyg_data.edge_attr is not None:
            obs_dict['edge_attr'] = pyg_data.edge_attr.numpy()
        else:
            # Need to provide edge_attr even if not used, matching space def
             num_edges = pyg_data.edge_index.shape[1]
             obs_dict['edge_attr'] = np.zeros((num_edges, config.NUM_EDGE_FEATURES), dtype=np.float32)


        # Return the PyG object directly, assuming SB3 Feature Extractor handles it.
        # This deviates from standard Gym API return types but matches project description.
        return pyg_data


    def _calculate_interval_reward(self):
        total_reward = 0.0
        for i in range(config.NUM_INTERSECTIONS):
            tls_id = config.NODE_IDX_TO_INTERSECTION_ID[i]
            incoming_lanes = state_builder._get_incoming_lanes(tls_id) # Use cached/internal logic

            current_queue_sum = 0
            current_wait_time_sum = 0
            for lane_id in incoming_lanes:
                current_queue_sum += sumo_connector.get_lane_halting_vehicle_count(lane_id)
                current_wait_time_sum += sumo_connector.get_lane_waiting_time_sum(lane_id)

            # Reward is negative of weighted congestion metrics
            reward_i = - (config.REWARD_QUEUE_WEIGHT * current_queue_sum +
                          config.REWARD_WAIT_TIME_WEIGHT * current_wait_time_sum)
            total_reward += reward_i

        return total_reward / config.NUM_INTERSECTIONS # Average reward per intersection


    def _get_info(self):
        # Provide auxiliary information, e.g., individual intersection rewards
        info = {
            "simulation_step": self._simulation_step_count,
            "episode_step": self._episode_step_count,
        }
        # Add detailed metrics if needed for logging/debugging
        # info["per_intersection_queue"] = ...
        # info["per_intersection_wait"] = ...
        return info

    def _handle_sim_error(self):
         # Called if sumo_connector.simulation_step() returns False
         print("Simulation error detected. Ending episode.")
         # Return dummy values consistent with Gym API
         # Observation should be valid based on last successful step or reset state
         obs = self._get_observation() # Get last valid state
         reward = config.GRIDLOCK_PENALTY # Penalize heavily
         terminated = True # End episode due to error
         truncated = True # Indicate abnormal termination
         info = self._get_info()
         info["error"] = "SUMO simulation connection lost or errored."
         sumo_connector.close_simulation() # Ensure cleanup
         return obs, reward, terminated, truncated, info


    def render(self):
        if self.render_mode == "human":
            # Assumes SUMO GUI is running if use_gui=True in constructor
            # No specific rendering code needed here unless adding custom overlays
            pass
        elif self.render_mode == "rgb_array":
            # This would require capturing the SUMO GUI image, complex setup
            return np.zeros((100, 100, 3), dtype=np.uint8) # Placeholder
        return None

    def close(self):
        sumo_connector.close_simulation()
        print("TrafficEnv closed.")