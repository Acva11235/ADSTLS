"""
This module handles the evaluation of a pre-trained GNN-DQN traffic signal 
control agent. It loads a specified agent model, runs it in the SUMO 
environment for a defined number of episodes using deterministic actions, 
collects performance metrics (like average waiting time, queue length), 
and reports the results. It serves to assess the agent's performance 
after training, potentially comparing it against baselines.
"""

import os
import time
import numpy as np
import pandas as pd
from stable_baselines3.common.monitor import Monitor

import config
from traffic_env import TrafficEnv # Environment definition
from agent import load_agent # Agent loading function
import sumo_connector # To potentially query final stats if needed

def evaluate_agent(model_path, num_episodes=10, use_gui=False):
    """
    Loads a trained agent and evaluates its performance over multiple episodes.

    Args:
        model_path (str): Path to the saved agent model (.zip file).
        num_episodes (int): Number of episodes to run for evaluation.
        use_gui (bool): Whether to run SUMO with the GUI for visualization.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    print(f"Evaluating model: {model_path}")
    print(f"Running for {num_episodes} episodes.")

    # --- Environment Setup ---
    # Use a single, monitored environment for evaluation
    eval_env = Monitor(TrafficEnv(use_gui=use_gui))
    print("Evaluation Environment Created.")

    # --- Load Agent ---
    agent = load_agent(model_path, env=eval_env) # Pass env for potential action space check
    if agent is None:
        eval_env.close()
        return
    print("Agent Loaded.")

    all_episode_rewards = []
    all_episode_lengths = []
    all_episode_avg_wait_times = []
    all_episode_avg_queue_lengths = []
    # Add other metrics as needed

    try:
        for episode in range(num_episodes):
            obs, info = eval_env.reset()
            terminated = False
            truncated = False
            episode_reward = 0
            episode_length = 0
            cumulative_wait_time = 0.0
            cumulative_queue_length = 0.0
            step_count = 0

            print(f"Starting Evaluation Episode {episode + 1}/{num_episodes}...")

            while not terminated and not truncated:
                # Use deterministic=True for evaluation
                action, _states = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)

                episode_reward += reward
                episode_length += 1
                step_count += 1

                # Accumulate metrics - requires access within the env or sumo_connector
                current_total_wait = 0
                current_total_queue = 0
                for i in range(config.NUM_INTERSECTIONS):
                    tls_id = config.NODE_IDX_TO_INTERSECTION_ID[i]
                    incoming_lanes = eval_env.get_wrapper_attr('state_builder')._get_incoming_lanes(tls_id) # Access via Monitor wrapper
                    for lane_id in incoming_lanes:
                        current_total_wait += sumo_connector.get_lane_waiting_time_sum(lane_id)
                        current_total_queue += sumo_connector.get_lane_halting_vehicle_count(lane_id)

                cumulative_wait_time += current_total_wait
                cumulative_queue_length += current_total_queue


                if (terminated or truncated):
                    print(f"Episode {episode + 1} finished after {episode_length} steps.")
                    print(f"  Total Reward: {episode_reward:.2f}")
                    # Use info from Monitor wrapper if available
                    if 'episode' in info:
                        print(f"  Monitor Reward: {info['episode']['r']:.2f}, Length: {info['episode']['l']}")
                        all_episode_rewards.append(info['episode']['r'])
                        all_episode_lengths.append(info['episode']['l'])
                    else:
                        # Fallback if Monitor info isn't directly available in last step's info
                        all_episode_rewards.append(episode_reward)
                        all_episode_lengths.append(episode_length)


                    avg_wait_time = cumulative_wait_time / step_count if step_count > 0 else 0
                    avg_queue_length = cumulative_queue_length / step_count if step_count > 0 else 0

                    print(f"  Avg Step Wait Time: {avg_wait_time:.2f}")
                    print(f"  Avg Step Queue Length: {avg_queue_length:.2f}")

                    all_episode_avg_wait_times.append(avg_wait_time)
                    all_episode_avg_queue_lengths.append(avg_queue_length)
                    # Reset episode accumulators (though loop restarts anyway)
                    break # Exit while loop for this episode

            # Short delay if GUI is used to allow viewing the end state
            if use_gui:
                time.sleep(1)

    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
    finally:
        eval_env.close()
        print("Evaluation environment closed.")

    # --- Report Results ---
    if all_episode_rewards:
        print("\n--- Evaluation Summary ---")
        mean_reward = np.mean(all_episode_rewards)
        std_reward = np.std(all_episode_rewards)
        mean_length = np.mean(all_episode_lengths)
        mean_wait_time = np.mean(all_episode_avg_wait_times)
        mean_queue_length = np.mean(all_episode_avg_queue_lengths)

        print(f"Number of Episodes: {len(all_episode_rewards)}")
        print(f"Average Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"Average Episode Length: {mean_length:.2f}")
        print(f"Average Step Waiting Time: {mean_wait_time:.2f}")
        print(f"Average Step Queue Length: {mean_queue_length:.2f}")

        # Optional: Save results to a file
        results_df = pd.DataFrame({
            'episode': list(range(1, len(all_episode_rewards) + 1)),
            'reward': all_episode_rewards,
            'length': all_episode_lengths,
            'avg_wait_time': all_episode_avg_wait_times,
            'avg_queue_length': all_episode_avg_queue_lengths
        })
        results_filename = f"evaluation_results_{os.path.basename(model_path).replace('.zip', '')}.csv"
        results_path = os.path.join(os.path.dirname(model_path), results_filename) # Save near the model
        try:
            results_df.to_csv(results_path, index=False)
            print(f"Evaluation results saved to: {results_path}")
        except Exception as e:
            print(f"Error saving evaluation results: {e}")

    else:
        print("No episodes completed successfully for evaluation.")


if __name__ == "__main__":
    # Example Usage: Replace with the actual path to your trained model
    # Assumes a model exists in the default save location structure
    # Find the latest run directory based on timestamp sorting
    model_parent_dir = config.MODEL_SAVE_DIR
    if os.path.exists(model_parent_dir):
        all_runs = sorted([d for d in os.listdir(model_parent_dir) if os.path.isdir(os.path.join(model_parent_dir, d))])
        if all_runs:
            latest_run_dir = os.path.join(model_parent_dir, all_runs[-1])
            # Look for the 'best_model.zip' or the final model
            best_model_file = os.path.join(latest_run_dir, "best_model", "best_model.zip")
            final_model_file = os.path.join(latest_run_dir, "dqn_gnn_model_final.zip")

            if os.path.exists(best_model_file):
                model_to_evaluate = best_model_file
            elif os.path.exists(final_model_file):
                model_to_evaluate = final_model_file
            else:
                print(f"Error: No suitable model (.zip) found in the latest run directory: {latest_run_dir}")
                model_to_evaluate = None
        else:
            print(f"Error: No run directories found in {model_parent_dir}")
            model_to_evaluate = None
    else:
        print(f"Error: Model directory does not exist: {model_parent_dir}")
        model_to_evaluate = None


    if model_to_evaluate:
        evaluate_agent(
            model_path=model_to_evaluate,
            num_episodes=config.EVALUATION_EPISODES, # Use config value for consistency
            use_gui=True # Set to True to watch the evaluation
        )
    else:
        print("Evaluation skipped as no model path was determined.")