import gym
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from tqdm import tqdm

from RL_algorithms.Torch.Imitation import core
from RL_algorithms.Torch.Imitation.memory import ReplayBuffer
import SpaceRobotEnv

from torch.utils.tensorboard import SummaryWriter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2

class ImitationDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.rgb_files = sorted(os.listdir(os.path.join(root_dir, 'rgb')))
        self.depth_files = sorted(os.listdir(os.path.join(root_dir, 'depth')))
        self.action_files = sorted(os.listdir(os.path.join(root_dir, 'expert_actions')))
        
    def __len__(self):
        return len(self.rgb_files)
    
    def __getitem__(self, idx):
        rgb_image = torch.load(os.path.join(self.root_dir, 'rgb', self.rgb_files[idx]))
        depth_files = torch.load(os.path.join(self.root_dir, 'depth', self.depth_files[idx]))
        action_files = torch.load(os.path.join(self.root_dir, 'expert_actions', self.action_files[idx]))
        
        return {'rgb_image': rgb_image, 'depth_image': depth_files, 'expert_actions': action_files}


    
    # Now you can use these batches in your training process


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)    

def test(env, model, num_test_episodes=5, max_ep_len=100):
    print("testing model")

    for j in tqdm(range(num_test_episodes)):

        observation, ep_len = env.reset(), 0

        while (ep_len < max_ep_len):

            image_tensor = torch.tensor(observation["rawimage"].transpose((2, 0, 1)), dtype=torch.uint8)
            image_tensor2 = image_tensor.clone().to(dtype=torch.float32)
            image_tensor2 = image_tensor2.reshape(1, 3, 64, 64)

            depth = observation["depth"].astype(np.float32)
            depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth)) * 255

            depth_tensor = torch.tensor(depth, dtype=torch.uint8)
            depth_tensor2 = depth_tensor.clone().to(dtype=torch.float32)
            depth_tensor2 = depth_tensor2.reshape(1, 1, 64, 64)

            actions = model(image_tensor2,depth_tensor2)

            try:
                actions =  actions.cpu().numpy()
            except:
                actions = actions.detach().numpy()
            
            observation, _, _, _, _ = env.step(actions.reshape(6,))

            distance_error = goal_distance(observation['achieved_goal'], observation['desired_goal'])
            
            writer.add_scalar("Testing - Distance/Error", distance_error, ep_len)

            if(distance_error < 0.05):
                break

            ep_len += 1
        

def imitate( writer, data_path, save_path, model_path, device=device, output_channels=6, seed=0, epochs=100, batch_size=128):

    print(f"\n DEVICE : {device} \n")

    torch.manual_seed(seed)
    np.random.seed(seed)
    img_size = 64

    env = gym.make('SpaceRobotImage-v0')

    imitate_agent = core.ImitationAgent(output_channels=output_channels, device=device).to(device)
    if model_path:
        imitate_agent.load_state_dict(torch.load(model_path, map_location=device))
        print(f"MODEL LOADED from {model_path}")
        imitate_agent.to(device)
    imitate_agent.train()

    criterion = nn.MSELoss()
    optimizer = Adam(imitate_agent.parameters(), lr=0.001)

    # Assuming you have data saved in 'data' directory
    dataset = ImitationDataset(root_dir= data_path)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("\n Starting Training \n")


    for epoch in range(epochs):
        torch.save(imitate_agent.state_dict(), f"{save_path}model{epoch}.pt")

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):

            img_obs, depth_obs, expert_actions = batch['rgb_image'], batch['depth_image'], batch['expert_actions']
        
            actions = imitate_agent(img_obs, depth_obs)

            actions.to(device)

            optimizer.zero_grad()
            # Compute loss between CNN actions and expert actions
            loss = criterion(actions, expert_actions)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            writer.add_scalar("Imitation Loss", loss, epoch)

        if (epoch>1) and ((epoch+1) % 10 == 0):
            imitate_agent.eval()
            test(env, imitate_agent, num_test_episodes=5, max_ep_len=100)
            imitate_agent.train()
            torch.save(imitate_agent.state_dict(), f"{save_path}model{epoch}.pt")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',  type=str, default=None)
    parser.add_argument('--save_path',  type=str, default="imitate_run/")
    parser.add_argument('--data_path',  type=str, default="data")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    writer = SummaryWriter(f"{args.save_path}imitation_logs")
    imitate(writer=writer, save_path=args.save_path, data_path=args.data_path, model_path=args.model_path, epochs=args.epochs, batch_size=args.batch_size)







