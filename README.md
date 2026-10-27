# Flappy Bird DQN

A DQN agent (PyTorch) that learns to play Flappy Bird.

## Setup

```
pip install -r requirements.txt
```

## Commands

| Command | What it does |
|---|---|
| `python game_flappy_bird.py` | Play yourself (SPACE to flap) |
| `python agent.py flappybird1 --train` | Train from scratch |
| `python agent.py flappybird1 --train --resume` | Keep training the saved model |
| `python agent.py flappybird1` | Watch the trained model play |
 
Stop training with Ctrl+C. The best model is saved to `runs/flappybird1.pt` as it goes.

## Files

- `agent.py` - training and watching
- `dqn.py` - the neural network
- `experience_replay.py` - replay memory
- `game_flappy_bird.py` - manual play
- `parameters.yaml` - hyperparameters
- `runs/` - saved models and logs

## Reward guide

About **+4.5 reward per pipe**. 50 = about 10 pipes, 100 = about 20 pipes, 1000 = solved (episode is capped there).
