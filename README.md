# Self-Driving Car Simulation (Behavioural Cloning)

## Setup
1. Install Python packages: `pip install -r requirements.txt`
2. Install simulator (CARLA) and start server.
3. Collect data: `python src/collect.py --out data/drive_log.csv`
4. Train: `python src/train.py`
5. Evaluate: `python src/eval.py --checkpoint checkpoints/model_epochX.pt`

## Structure
- src/: code
- data/: images + csv
- checkpoints/: saved models
- logs/: tensorboard logs

## Extensions
- Add segmentation/object detection modules
- Switch to RL (PPO) for closed-loop control
