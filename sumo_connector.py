"""
This module acts as an interface layer for interacting with the SUMO 
(Simulation of Urban MObility) traffic simulator using the TraCI 
(Traffic Control Interface) API. It encapsulates the logic for starting, 
stepping through, and stopping the simulation, as well as querying various 
state information (vehicle data, traffic light status) and sending control 
commands (setting traffic light phases). The purpose is to abstract the 
low-level TraCI calls, providing a cleaner and more focused API for the 
rest of the project, particularly the RL environment (`traffic_env`).
"""

import traci
import traci.constants as tc
import sumolib
import sys
import os
import subprocess
import time

from config import (SUMO_EXECUTABLE, SUMO_CONFIG_FILE, NET_FILE, ROUTE_FILE, ADDITIONAL_FILES,
                    SIMULATION_STEP_LENGTH, INTERSECTION_IDS)


SUMO_CMD = None
_connection = None
_sumo_proc = None

def _init_sumo_cmd(use_gui):
    global SUMO_CMD
    executable = SUMO_EXECUTABLE if use_gui else "sumo"
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("Please declare environment variable 'SUMO_HOME'")

    SUMO_CMD = [
        executable,
        "-c", SUMO_CONFIG_FILE,
        "--net-file", NET_FILE,
        "--step-length", str(SIMULATION_STEP_LENGTH),
        "--time-to-teleport", "-1", 
        "--collision.action", "remove", #
        "--no-warnings", "true", 
        "--seed", "42", # 
        "--remote-port", str(sumolib.miscutils.getFreeSocketPort())
    ]

def start_simulation(use_gui=False):
    global _connection, _sumo_proc
    if _connection is not None:
        print("Warning: Simulation already running.")
        return _connection

    _init_sumo_cmd(use_gui)
    port = int(SUMO_CMD[SUMO_CMD.index("--remote-port") + 1])

    try:
        # Using traci.start directly
        traci.start(SUMO_CMD, port=port, label="sim1")
        _connection = traci.getConnection("sim1")
        # Alternative: subprocess for more control, but traci.start is simpler
        # _sumo_proc = subprocess.Popen(SUMO_CMD)
        # time.sleep(2) # Allow time for SUMO to start
        # _connection = traci.connect(port=port, label="sim1")

    except traci.TraCIException as e:
        print(f"Error starting SUMO: {e}")
        print(f"Command used: {' '.join(SUMO_CMD)}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during SUMO start: {e}")
        print(f"Command used: {' '.join(SUMO_CMD)}")
        sys.exit(1)

    return _connection


def close_simulation():
    global _connection, _sumo_proc
    if _connection:
        try:
            traci.close()
        except traci.TraCIException as e:
            print(f"Error closing TraCI connection: {e}")
        finally:
            _connection = None
    if _sumo_proc:
        _sumo_proc.terminate()
        try:
            _sumo_proc.wait(timeout=10) 
        except subprocess.TimeoutExpired:
            _sumo_proc.kill() 
        _sumo_proc = None

def simulation_step():
    if _connection:
        try:
            _connection.simulationStep()
            return True
        except traci.TraCIException as e:
            print(f"Error during simulation step: {e}. Closing simulation.")
            close_simulation()
            return False
    else:
        print("Error: Simulation not running.")
        return False

def get_current_time():
    if _connection:
        return _connection.simulation.getTime()
    return 0

def get_network_data():
    if _connection:
        net = sumolib.net.readNet(NET_FILE)
        return net
    return None

# --- Traffic Light Queries ---

def get_traffic_light_ids():
    if _connection:
        return _connection.trafficlight.getIDList()
    return []

def get_controlled_lanes(tls_id):
    if _connection:
        return _connection.trafficlight.getControlledLanes(tls_id)
    return []

def get_incoming_lanes(tls_id, net):
    incoming_lanes = set()
    tls_node = net.getNode(tls_id)
    for edge in tls_node.getIncoming():
        # Check if the edge connects two internal intersections or comes from boundary
        # This logic might need refinement depending on how boundary edges are handled
        # For now, assume all incoming edges to a controlled TLS are relevant
        for lane in edge.getLanes():
            incoming_lanes.add(lane.getID())
    return list(incoming_lanes)


def get_traffic_light_phase(tls_id):
    if _connection:
        try:
            return _connection.trafficlight.getPhase(tls_id)
        except traci.TraCIException as e:
            print(f"Error getting phase for {tls_id}: {e}")
            return -1 # Indicate error
    return -1

def get_traffic_light_phase_duration(tls_id):
    if _connection:
        try:
                return _connection.trafficlight.getPhaseDuration(tls_id)
        except traci.TraCIException as e:
            print(f"Error getting phase duration for {tls_id}: {e}")
            return -1
        return -1

def get_time_since_last_phase_switch(tls_id):
    if _connection:
        try:
            # SUMO reports time since the start of the *current* phase
            # Note: getPhaseDuration is total duration, not elapsed
            # We need getNextSwitch - getTime() for time *until* next switch
            # Time *since* last switch requires tracking externally or using this value
            # correctly. Let's return time spent in current step's phase.
            # Be careful: This might reset to 0 briefly during yellow phases.
            # A more robust way involves storing the last switch time externally.
            return _connection.trafficlight.getPhaseDuration(tls_id) - \
                        (_connection.trafficlight.getNextSwitch(tls_id) - _connection.simulation.getTime())

        except traci.TraCIException as e:
            print(f"Error getting next switch time for {tls_id}: {e}")
            return 0.0
    return 0.0


# --- Lane/Edge Queries ---

def get_lane_halting_vehicle_count(lane_id):
    if _connection:
        try:
            return _connection.lane.getLastStepHaltingNumber(lane_id)
        except traci.TraCIException as e:
            # Lane might not exist if vehicle just left, handle gracefully
            # print(f"Warning: Error getting halting count for lane {lane_id}: {e}")
            return 0
    return 0

def get_lane_waiting_time_sum(lane_id):
    if _connection:
        try:
            # Calculate sum manually as TraCI doesn't have a direct sum function for lane waiting time
            vehicle_ids = _connection.lane.getLastStepVehicleIDs(lane_id)
            total_wait_time = 0.0
            for veh_id in vehicle_ids:
                total_wait_time += _connection.vehicle.getWaitingTime(veh_id)
            return total_wait_time
        except traci.TraCIException as e:
            # print(f"Warning: Error getting waiting time sum for lane {lane_id}: {e}")
            return 0.0
    return 0.0


def get_edge_vehicle_count(edge_id):
    if _connection:
        try:
            return _connection.edge.getLastStepVehicleNumber(edge_id)
        except traci.TraCIException as e:
            # print(f"Warning: Error getting vehicle count for edge {edge_id}: {e}")
            return 0
    return 0

def get_edge_mean_speed(edge_id):
    if _connection:
        try:
            return _connection.edge.getLastStepMeanSpeed(edge_id)
        except traci.TraCIException as e:
            # print(f"Warning: Error getting mean speed for edge {edge_id}: {e}")
            return 0.0
    return 0.0

def get_edge_density(edge_id):
    if _connection:
        try:
            # Density = veh_count / length
            count = _connection.edge.getLastStepVehicleNumber(edge_id)
            length = _connection.edge.getLength(edge_id)
            return count / length if length > 0 else 0.0
        except traci.TraCIException as e:
            # print(f"Warning: Error calculating density for edge {edge_id}: {e}")
            return 0.0
    return 0.0

def get_edge_travel_time(edge_id):
    if _connection:
        try:
            # Travel time can be estimated or read directly if configured
            return _connection.edge.getTravelTime(edge_id)
        except traci.TraCIException as e:
            # print(f"Warning: Error getting travel time for edge {edge_id}: {e}")
            return 0.0
    return 0.0

# --- Control Commands ---

def set_traffic_light_phase(tls_id, phase_index):
    if _connection:
        try:
            _connection.trafficlight.setPhase(tls_id, phase_index)
        except traci.TraCIException as e:
            print(f"Error setting phase for {tls_id} to {phase_index}: {e}")