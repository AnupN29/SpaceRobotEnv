import gym
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from tqdm import tqdm

from RL_algorithms.Torch.Imitation import core
from RL_algorithms.Torch.Imitation.memory import ReplayBuffer
from RL_algorithms.Torch.SAC.SAC_ENV.core import MLPActorCritic
import SpaceRobotEnv

from torch.utils.tensorboard import SummaryWriter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


writer = SummaryWriter("/content/drive/MyDrive/SLP419_RL_Project/imitation_logs")

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)    

def collect_data(env, expert_model, replay_buffer):
    

    while replay_buffer.size < replay_buffer.max_size:

        observation = env.reset()

        for i in range(100): 
            
            image_tensor = torch.tensor(observation["rawimage"].transpose((2, 0, 1)), dtype=torch.uint8)
            image_tensor2 = image_tensor.clone().to(dtype=torch.float32)

            depth = observation["depth"].astype(np.float32)
            depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth)) * 255
            depth_tensor = torch.tensor(depth, dtype=torch.uint8)
            depth_tensor2 = depth_tensor.clone().to(dtype=torch.float32)
            depth_tensor2 = depth_tensor2.reshape(1, 64, 64)

            expert_actions = expert_model.act(obs=torch.tensor(observation["observation"], dtype=torch.float32)) 
            replay_buffer.store([image_tensor2, depth_tensor2], expert_actions)
            
            observation, _, _, _, _ = env.step(expert_actions)

            distance_error = goal_distance(observation['achieved_goal'], observation['desired_goal'])
            if(distance_error < 0.05 or i == 100):
                break


def imitate(expert_model_path, device=device, output_channels=6, seed=0, replay_size=int(1e5), epochs=100, batch_size=128, writer=writer):

    print(f"\n DEVICE : {device} \n")

    torch.manual_seed(seed)
    np.random.seed(seed)
    img_size = 64

    env = gym.make('SpaceRobotImage-v0')

    imitate_agent = core.ImitationAgent(output_channels=output_channels, device=device).to(device)
    expert_agent = MLPActorCritic(env.observation_space['observation'], env.action_space, hidden_sizes=[256]*2).to(device)

    if torch.cuda.is_available():
        expert_agent.load_state_dict(torch.load(expert_model_path))
    else:
        expert_agent.load_state_dict(torch.load(expert_model_path,map_location=torch.device('cpu'))) 
    expert_agent.eval()
    expert_agent.to(device)

    action_dim = env.action_space.shape[0]

    replay_buffer = ReplayBuffer(obs_dim = img_size, act_dim = action_dim, size=replay_size)
    collect_data(env, expert_agent, replay_buffer)

    criterion = nn.MSELoss()
    optimizer = Adam(imitate_agent.parameters(), lr=0.001)

    for epoch in tqdm(range(epochs)):

        batch = replay_buffer.sample_batch(batch_size)
        img_obs, depth_obs, expert_actions = batch['img_obs'], batch['depth_obs'], batch['expert_act']
    
        actions = imitate_agent(img_obs, depth_obs)

        optimizer.zero_grad()
        # Compute loss between CNN actions and expert actions
        loss = criterion(actions, expert_actions)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        writer.add_scalar("Imitation Loss", loss, epoch)

    torch.save(imitate_agent.state_dict(), f"/content/drive/MyDrive/SLP419_RL_Project/imitate_model/model.pt")





