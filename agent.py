"""Train or watch a DQN agent.

Train:  python agent.py flappybird1 --train     (Ctrl+C to stop; best model is saved as you go)
Resume: python agent.py flappybird1 --train --resume   (starts from the saved model at epsilon_min)
Watch:  python agent.py flappybird1
"""
import argparse
import os
import random
from datetime import datetime
from itertools import count

import flappy_bird_gymnasium  # noqa: F401 - registers "FlappyBird-v0"
import gymnasium as gym
import torch
import yaml
from torch import nn

from dqn import DQN
from experience_replay import ReplayMemory

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"



class Agent:
    def __init__(self, hyperparameter_set):
        #self.hyperparameter_set = hyperparameter_set
        
        with open("parameters.yaml", "r") as f:
            params = yaml.safe_load(f)[hyperparameter_set]

        self.env_id = params["env_id"]
        self.env_make_params = params.get("env_make_params", {})
        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]
        self.epsilon_init = params["epsilon_init"]
        self.epsilon_decay = params["epsilon_decay"]
        self.epsilon_min = params["epsilon_min"]
        self.network_sync_rate = params["network_sync_rate"]
        self.learning_rate = params["learning_rate"]
        self.discount_factor = params["discount_factor"]
        self.hidden_dim = params["hidden_dim"]
        self.stop_on_reward = params["stop_on_reward"]

        self.model_file = os.path.join(RUNS_DIR, f"{hyperparameter_set}.pt")
        self.log_file = os.path.join(RUNS_DIR, f"{hyperparameter_set}.log")
        self.loss_fn = nn.MSELoss()

    def run(self, is_training=True, render=False, resume=False):
        
        env = gym.make(self.env_id, render_mode="human" if render else None, **self.env_make_params)
        
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states, num_actions, self.hidden_dim).to(device)

        if is_training:
            if resume:
                # continue from the saved policy: skip exploration and keep its best reward
                checkpoint = torch.load(self.model_file, weights_only=True)
                policy_dqn.load_state_dict(checkpoint["model"])
                best_reward = checkpoint["best_reward"]
                epsilon = self.epsilon_min 
            else:
                # a fresh run would overwrite the saved model with its first (bad) episode
                if os.path.exists(self.model_file):
                    raise SystemExit(f"{self.model_file} already exists. Use --resume to keep training it, "
                                     f"or delete/rename it to start from scratch.")
                best_reward = float("-inf")
                epsilon = self.epsilon_init

            memory = ReplayMemory(self.replay_memory_size)

            target_dqn = DQN(num_states, num_actions, self.hidden_dim).to(device)

            target_dqn.load_state_dict(policy_dqn.state_dict())
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)

            steps_since_sync = 0
            self.log(f"Training {self.env_id} on {device} (resume={resume}, epsilon={epsilon})")
        else:
            # load best policy from file and set epsilon=0 (no exploration)
            policy_dqn.load_state_dict(torch.load(self.model_file, weights_only=True)["model"])
            policy_dqn.eval()
            epsilon = 0

        for episode in count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            
            terminated = False
            truncated = False
            episode_reward = 0.0

            while not (terminated or truncated) and episode_reward < self.stop_on_reward:
                # Epsilon-greedy: explore randomly sometimes, otherwise pick the best known action
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(0)).argmax().item()

                new_state, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)

                if is_training:
                    memory.append((state, action, new_state, reward, terminated))
                    if len(memory) >= self.mini_batch_size:
                        self.optimize(memory.sample(self.mini_batch_size), policy_dqn, target_dqn)

                    steps_since_sync += 1
                    if steps_since_sync >= self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        steps_since_sync = 0

                state = new_state

            if not is_training:
                print(f"Episode {episode}: reward {episode_reward:.1f}, score {info.get('score', '-')}")
                continue

            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save({"model": policy_dqn.state_dict(), "best_reward": best_reward}, self.model_file)
                self.log(f"Episode {episode}: new best reward {best_reward:.1f} (epsilon {epsilon:.3f}) - model saved")

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, new_states, rewards, terminations = zip(*mini_batch)
        
        states = torch.stack(states)
        new_states = torch.stack(new_states)
        actions = torch.tensor(actions, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        terminations = torch.tensor(terminations, dtype=torch.float, device=device)

        # Target: reward + discounted best future Q-value (no future if the episode ended)
        with torch.no_grad():
            best_next_q = target_dqn(new_states).max(dim=1).values
            target_q = rewards + (1 - terminations) * self.discount_factor * best_next_q

        # Q-value the policy network gave to the action that was actually taken
        current_q = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def log(self, message):
        line = f"{datetime.now():%H:%M:%S} {message}"
        print(line)
        with open(self.log_file, "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or watch a DQN agent.")
    parser.add_argument("hyperparameters", help="section name in parameters.yaml, e.g. flappybird1")
    parser.add_argument("--train", action="store_true", help="train instead of watching the saved model")
    parser.add_argument("--resume", action="store_true", help="with --train: continue from the saved model, no exploration")
    args = parser.parse_args()

    agent = Agent(args.hyperparameters)
    if args.train:
        agent.run(is_training=True, resume=args.resume)
    else:
        agent.run(is_training=False, render=True)
