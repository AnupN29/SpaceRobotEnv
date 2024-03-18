from RL_algorithms.Torch.SAC.SAC_Image import cnn_model as cnn_core
from RL_algorithms.Torch.SAC.SAC_ENV import core
import gym
import torch
import torch.nn as nn


from torch.optim import Adam
import numpy as np
import SpaceRobotEnv
from torch.utils.tensorboard import SummaryWriter

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

def act(model, obs, deterministic=False):
    a, _ = model(obs, deterministic, False)
    return a.detach().numpy()

def update(data, n_update_step, criterion, optimizer):
    
    actions, expert_actions = data['action'], data['expert_action']

    optimizer.zero_grad()
    # Compute loss between CNN actions and expert actions
    loss = criterion(actions, expert_actions)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    writer.add_scalar("Imitation Loss", loss, n_update_step)

        

def train(writer, epochs, steps_per_epoch, env, expert_model, cnn_model, criterion, optimizer, replay_buffer, batch_size,device):
    cnn_model.train()
    n_update_step  = 0
    for ep in range(epochs):
        writer = SummaryWriter(f"logs/imitate")

        observation = env.reset()

        for i in range(steps_per_epoch):
            
            

            actions = act(model=cnn_model,obs=torch.tensor(observation["rawimage"].reshape(1, 3, 64, 64), dtype=torch.float32, device=device))
            actions = actions.reshape(6,) 
            expert_actions = expert_model.act(obs=torch.tensor(observation["observation"], dtype=torch.float32)) 

            # print(f"Student action : {actions.shape}")
            # print(f"Expert action : {expert_actions.shape}")
            
            observation, _, _, _, _ = env.step(expert_actions)

            if ep > 4:
                batch = replay_buffer.sample_batch(batch_size)
                update(batch, n_update_step, criterion, optimizer, device)
                n_update_step += 1
            
            replay_buffer.store(actions, expert_actions)

            
            # print("Distance/Error :", distance_error)

            distance_error = goal_distance(observation['achieved_goal'], observation['desired_goal'])
            if(distance_error < 0.05 or i == steps_per_epoch):
                break
        if ep > 4:
            torch.save(cnn_model.state_dict(), f"model_epoch_{ep}.pt")
            test(writer, env, cnn_model, ep)

def test(writer, env, cnn_model,n):
    cnn_model.eval()
    for ep in range(5):
        writer = SummaryWriter(f"logs/imitate/test{n}/dist_err{ep}")

        observation = env.reset()
        for i in range(100):
            actions = act(model=cnn_model,obs=torch.tensor(observation["rawimage"].reshape(1, 3, 64, 64), dtype=torch.float32)).detach().numpy()

            actions = actions.reshape(6,) 

            observation, _, _, _, _ = env.step(actions)

            distance_error = goal_distance(observation['achieved_goal'], observation['desired_goal'])

            writer.add_scalar("Distance/Error", distance_error, i)
            # print("Distance/Error :", distance_error)

            if(distance_error < 0.05 or i == 100):
                break



def imitate(CNNmodel_path=None, CNNactor=cnn_core.CNNActor, model_path=None, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=400, epochs=100, replay_size=int(1e6), batch_size=128,writer=None):
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    env = gym.make('SpaceRobotImage-v0')
    action_dim = env.action_space.shape[0]


    model_cnn = CNNactor(act_dim=env.action_space.shape[0], activation=nn.ReLU, act_limit=env.action_space.high[0], device=device, **ac_kwargs).to(device)

    if CNNmodel_path:
        model_cnn.load_state_dict(torch.load(CNNmodel_path))
        print(f"MODEL LOADED from {CNNmodel_path}")
    model_cnn.train()

    expert_model = actor_critic(env.observation_space['observation'], env.action_space, **ac_kwargs).to(device)

    if model_path:
        expert_model.load_state_dict(torch.load(model_path))
        print(f"MODEL LOADED from {model_path}")
        
    for param in expert_model.parameters():
        param.requires_grad = False

    criterion = nn.MSELoss()
    optimizer = Adam(model_cnn.parameters(), lr=0.001)

    replay_buffer = cnn_core.ReplayBuffer(act_dim=action_dim, size=replay_size)

    train(writer, epochs, steps_per_epoch, env, expert_model, model_cnn, criterion, optimizer, replay_buffer, batch_size,device=device)

    # Close the environment
    env.close()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--replay_size', type=int, default=int(1e6))
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    # from spinup.utils.run_utils import setup_logger_kwargs
    # logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    logger_kwargs = None

    torch.set_num_threads(torch.get_num_threads())
    writer = SummaryWriter("logs")
    
    imitate(epochs=args.epochs, seed=args.seed, ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), writer=writer)

