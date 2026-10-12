# masters_thesis_2027

Master's thesis project. The code is based on Henrik Fjellheim's master's thesis
"Short-Term Trajectory Planning for a Non-Holonomic Robot Car: Utilizing Reinforcement Learning
in conjunction with a Predefined Vehicle Model" (NTNU, 2023).
Original repo: https://github.com/henrfj/master_thesis_2023

![PDRL architecture](figures/PDRL.png)

## Setup

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Runs on CPU by default. To use an NVIDIA GPU, set `disable_cuda=False` in `DDPG/ddpg_torch.py`.
Run all commands from the repo root.

## Running

Drive the car with the arrow keys (Henrik's keyboard mode):

```
python -c "from DDPG.visualize_ddpg import visualize_v23; visualize_v23(folder='DDPG/checkpoints/v23', user_controlled=True)"
```

Up/down is throttle and reverse, left/right is steering. Close the window to quit.
This needs torch, since the trained agent is loaded.

Simulation with only rule-based drivers:

```
python visualization.py          # Henrik's "boats" demo, using the car model
python visualization.py cars     # cars in a city block map
```

Trained agents:

```
python -m DDPG.visualize_ddpg pdrl
python -m DDPG.visualize_ddpg pdrl_dynamic
python -m DDPG.visualize_ddpg ddpg
python -m DDPG.visualize_ddpg ddpg_dynamic
```

Training:

```
python -m DDPG.main_torch pdrl [episodes]
python -m DDPG.main_torch ddpg [episodes]
python -m DDPG.main_torch ddpg_dynamic [episodes]
```

Weights are saved to `DDPG/checkpoints/my_*` and plots to `DDPG/plots/`.

## Checkpoints

`v22`, `v22_naples_nn` and `v23` are Henrik's trained weights for the car. `v40_fw` is also from
his repo, but it is not used by any of the commands above and contains the same weights as `v22`.

## Files

```
Object.py           obstacles (polygons)
Vehicle.py          car model and circograms (simulated lidar)
Agent.py            rule-based driver for the other vehicles
limo.py             pairs a Vehicle with an Agent
visualization.py    pygame drawing, maps and demos
DDPG/
  ddpg_torch.py     DDPG networks, Agent (DDPG) and MPC_Agent (PDRL)
  environments.py   gym environments
  main_torch.py     training
  visualize_ddpg.py running trained agents and keyboard mode
  utils.py          learning curve plot
  checkpoints/      trained weights
```

Environments in `DDPG/environments.py`:

- `ClosedField_v22`: DDPG, static maps (`four_walls`, `naples_street`)
- `ClosedField_v23_dyna`: DDPG, open area with 6 rule-based vehicles
- `MPC_environment_v40`: PDRL, static maps
- `MPC_environment_v41`: PDRL with rule-based vehicles

## Changes from Henrik's repo

- Removed `main.py`, `Testing.py`, the notebooks, `dev_ideas.md`, the A* planner, environments
  v00 to v21 and their training and visualization functions, `ST_approximator.py`,
  `data_collection.py`, the training data, unused methods, and most checkpoints and figures.
- `Vehicle.py`: fixed the state indexing so it runs with numpy 2.
- Checkpoint folders are created when saving.
- `main_torch.py`, `visualize_ddpg.py` and `visualization.py` take the mode as a command line
  argument instead of commenting function calls in and out.
