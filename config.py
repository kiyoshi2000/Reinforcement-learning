import pickle
import gymnasium as gym
import highway_env
import time


config_dict = {
    "observation": {
        "type": "OccupancyGrid",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20],
        },
        "grid_size": [[-20, 20], [-20, 20]],
        "grid_step": [5, 5],
        "absolute": False,
    },
    "action": {
        "type": "DiscreteMetaAction",
    },
    "lanes_count": 4,
    "vehicles_count": 10,
    "duration": 60,  # [s]
    "initial_spacing": 2,
    "collision_reward": -5,  # The reward received when colliding with a vehicle.
    "right_lane_reward": 0.5,  # The reward received when driving on the right-most lanes, linearly mapped to
    # zero for other lanes.
    "high_speed_reward": 0.1,  # The reward received when driving at full speed, linearly mapped to zero for
    # lower speeds according to config["reward_speed_range"].
    "lane_change_reward": 0,
    "reward_speed_range": [
        20,
        30,
    ],  # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD].
    "simulation_frequency": 5,  # [Hz]
    "policy_frequency": 5,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # [px]
    "screen_height": 150,  # [px]
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": True,
    "render_agent": True,
    "offscreen_rendering": False,
    "disable_collision_checks": False,
}



# Save the configuration 
with open("config.pkl", "wb") as f:
    pickle.dump(config_dict, f)

if (0):
    # Create and configure the environment
    env = gym.make("highway-fast-v0", render_mode="rgb_array")
    env.unwrapped.configure(config_dict)

    # Initialize the environment
    obs, info = env.reset()

    # Run 100 steps with random actions and render
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        time.sleep(0.1)

    env.close()
