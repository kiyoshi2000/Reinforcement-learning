# Reinforcement Learning – Final Project (IA 2024–25)

This repository contains the code and artifacts for a 4-part reinforcement learning project, combining custom implementations and baseline agents using environments from `highway-env`.

## Project Structure

### Task 1 — DQN From Scratch  
**Objective**: Train a DQN agent on a pre-defined environment using a custom configuration.  
- `task1.ipynb`: DQN implementation and training code.  
- `config.py`: Custom configuration file used to define the environment parameters.  
- `config.pkl`: Serialized config used for reproducibility.  
- `trained_dqn_agent.pth`: Saved model weights.

### Task 2 — Continuous Actions  
**Objective**: Choose an environment, enable continuous control, and implement a policy-gradient-based method.  
- `task2.ipynb`: Custom implementation using REINFORCE with baseline on a modified racetrack environment.  
- `actor_model.pth`, `critic_model.pth`: Trained actor and critic networks for the continuous agent.

### Task 3 — Stable-Baselines Agent  
**Objective**: Use an off-the-shelf algorithm from the `stable-baselines3` library on an environment of your choice.  
- `task3.ipynb`: Training pipeline using SAC in `parking-v0`.  
- `models/`: Contains SAC models and replay buffer saved during training.

### Task 4 — Extra Experiment  
**Objective**: Conduct a custom experiment related to RL learning behavior or generalization.  
- `task4.ipynb`: Analysis of safety, hyperparameter sensitivity, or transferability (depending on the question explored).  
- Results discussed and visualized inside the notebook.

## Notes

- The project is modular: each task can be run independently.
- All models were trained and tested using environments from `highway-env`.
- Code includes performance optimizations like disabling rendering and logging metrics for evaluation.
