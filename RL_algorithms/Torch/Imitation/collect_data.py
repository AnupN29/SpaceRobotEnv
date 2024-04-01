import torch
import numpy as np
import gym

from tqdm import tqdm
import os

import SpaceRobotEnv
from RL_algorithms.Torch.SAC.SAC_ENV.core import MLPActorCritic


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_data(rgb_tensor, depth_tensor, expert_actions_tensor, i, save_path):

    torch.save(expert_actions_tensor, f'{save_path}/expert_actions/sample_{i}.pt')
    torch.save(depth_tensor, f'{save_path}/depth/sample_{i}.pt')
    torch.save(rgb_tensor, f'{save_path}/rgb/sample_{i}.pt')


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)  

def collect_data(env, expert_model, max_samples, save_path):
    num_samples = 0
    os.makedirs(f'{save_path}/rgb', exist_ok=True)
    os.makedirs(f'{save_path}/depth', exist_ok=True)
    os.makedirs(f'{save_path}/expert_actions', exist_ok=True)
    with tqdm() as pbar:
        while num_samples < max_samples:

            observation = env.reset()

            for i in range(100): 
                
                image_tensor = torch.tensor(observation["rawimage"].transpose((2, 0, 1)), dtype=torch.uint8)
                image_tensor2 = image_tensor.clone().to(dtype=torch.float32)

                depth = observation["depth"].astype(np.float32)
                depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth)) * 255
                depth_tensor = torch.tensor(depth, dtype=torch.uint8)
                depth_tensor2 = depth_tensor.clone().to(dtype=torch.float32)
                depth_tensor2 = depth_tensor2.reshape(1, 64, 64)

                expert_actions = expert_model.act(obs=torch.tensor(observation["observation"], dtype=torch.float32, device=device)) 
                
                save_data(image_tensor2, depth_tensor2, torch.tensor(expert_actions), num_samples, save_path)
                num_samples+=1
                pbar.update(1)

                observation, _, _, _, _ = env.step(expert_actions)

                distance_error = goal_distance(observation['achieved_goal'], observation['desired_goal'])
                if(distance_error < 0.05 or i == 100):
                    break



def main(expert_model_path, device=device, save_path="data", seed=0, max_samples=int(1e5)):

    print(f"\n DEVICE : {device} \n")

    torch.manual_seed(seed)
    np.random.seed(seed)
    

    env = gym.make('SpaceRobotImage-v0')

    expert_agent = MLPActorCritic(env.observation_space['observation'], env.action_space, hidden_sizes=[256]*2).to(device)

    if torch.cuda.is_available():
        expert_agent.load_state_dict(torch.load(expert_model_path))
    else:
        expert_agent.load_state_dict(torch.load(expert_model_path,map_location=torch.device('cpu'))) 
    expert_agent.eval()
    expert_agent.to(device)

    print("\n Collecting Episodes \n")
    collect_data(env, expert_agent, max_samples, save_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_model',  type=str, default=None)
    parser.add_argument('--save_path',  type=str, default="data")
    parser.add_argument('--max_samples', type=int, default=int(1e5))
    args = parser.parse_args()

    main(expert_model_path=args.expert_model, save_path=args.save_path, max_samples=args.max_samples)