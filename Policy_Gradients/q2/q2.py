import gym
import numpy as np
import matplotlib.pyplot as plt
import pybullet
import torch
import torch.nn as nn
import torch.distributions as dist
from torch.distributions import MultivariateNormal
import time

#parameters
maxepisode = 500
maxiter = 200
gamma = 0.9
render_flag = True
opt_lr = 0.01
model_input_size = 4
model_output_size = 2
part = 3
device = "cuda"
pi_path = 'pi_ep_30.model'
optimizer_path = 'optimizer_ep_30.model'

class Pi(nn.Module):
    def __init__(self):
        super(Pi, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(8, 64),nn.ReLU(),
            nn.Linear(64, 16), nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        action_val = self.fc(x.float())
        return action_val

def get_return(reward_list):
    new_reward_list = []
    for ep in range(maxepisode):
        cur_ep_reward_list = []
        for cur_t in range(len(reward_list[ep])):
            cur_reward = 0
            for t in range(cur_t, len(reward_list[ep])):
                cur_reward += gamma ** (t-cur_t) * reward_list[ep][t]
            cur_ep_reward_list.append(cur_reward)
        cur_ep_reward_list = torch.from_numpy(np.array(cur_ep_reward_list)).to(device)
        new_reward_list.append(cur_ep_reward_list)
    return new_reward_list

def calculate_loss(return_list, log_prob_list):
    cur_policy_loss = 0
    if part == 3 or part == 4:
        all_returns = torch.cat(return_list).to(device)
        b = all_returns.mean()
        std = all_returns.std()

    for ep in range(maxepisode):
        for t in range(len(return_list[ep])):
            # Update return with different methods
            cur_return = 0
            if part == 1:
                cur_return = return_list[ep][0]
            elif part == 2:
                cur_return = return_list[ep][t]
            elif part == 3 or part == 4:
                cur_return = return_list[ep][t]
                cur_return = (cur_return - b) / std

            cur_policy_loss += (cur_return * (-log_prob_list[ep][t]))

    cur_policy_loss /= maxepisode

    return cur_policy_loss


def execute():
    print(part)
    env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=False)

    return_list = []
    pi = Pi().to(device)
    optimizer = torch.optim.Adam(pi.parameters(), lr=opt_lr)
    avg_reward_list =[]

    for i in range(maxiter):
        total_reward = 0
        log_prob_list = []
        reward_list = []

        for j in range(maxepisode):
            cur_reward_list = []
            cur_log_prob_list = []

            #execute episode
            episode_flag = False
            state = env.reset()

            while episode_flag == False:
                state = torch.from_numpy(state).to(device)
                action_val = pi(state)

                # choose the better action and get log of action prob
                action_dist = MultivariateNormal(action_val, 0.1*torch.eye(2).to(device))
                action = action_dist.sample()
                log_action_prob = action_dist.log_prob(action)
                state, reward, episode_flag, _ = env.step(action.cpu().numpy())

                # save rollouts
                cur_reward_list.append(reward)
                cur_log_prob_list.append(log_action_prob)

            total_reward += sum(cur_reward_list)
            reward_list.append(cur_reward_list)
            log_prob_list.append(cur_log_prob_list)
            #end of episode

        return_list = get_return(reward_list)
        cur_policy_loss = calculate_loss(return_list, log_prob_list)

        #save avg reward/episode and loss
        avg_reward_list.append(total_reward/maxepisode)

        print("Iter", i, "avg reward", avg_reward_list[-1])

        # policy_gradient_update
        optimizer.zero_grad()
        cur_policy_loss.backward()
        optimizer.step()
        #end of iteration

    #save result
    output = np.array(avg_reward_list)
    np.save("./q2"+ "_ep_" + str(maxepisode), output)
    torch.save(Pi.state_dict(), 'pi_ep_'+str(maxepisode)+'.model')
    torch.save(optimizer.state_dict(), 'optimizer_ep_'+str(maxepisode)+'.model')
    print("End: part " + str(part))

def render():
    env = gym.make("modified_gym_env:ReacherPyBulletEnv-v1", rand_init=False)
    env.render()
    state = env.reset()
    episode_flag = False

    #load models
    pi = Pi().to(device)
    optimizer = torch.optim.Adam(pi.parameters(), lr=opt_lr)
    pi_model = torch.load(pi_path)
    optimizer_model = torch.load(optimizer_path)
    pi.load_state_dict(pi_model)
    optimizer.load_state_dict(optimizer_model)

    while episode_flag == False:
        state = torch.from_numpy(state).to(device)
        action_val = pi(state)

        # choose the better action and get log of action prob
        action_dist = MultivariateNormal(action_val, 0.1 * torch.eye(2).to(device))
        action = action_dist.sample()
        state, reward, episode_flag, _ = env.step(action.cpu().numpy())
        time.sleep(0.2)

def main():
    global part
    global maxepisode

    if render_flag == True:
        render()
        return

    episode_lists = [30, 60, 100]
    for ep in episode_lists:
        maxepisode = ep
        print(maxepisode)
        execute()

if __name__ == '__main__':
    main()