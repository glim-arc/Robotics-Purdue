import gym
import numpy as np
import pybullet
import torch
import torch.nn as nn
import torch.distributions as dist

#parameters
maxepisode = 500
maxiter = 200
gamma = 0.99
render_flag = False
opt_lr = 0.01
model_input_size = 4
model_output_size = 2
part = 3
device = "cpu"

class Pi(nn.Module):
    def __init__(self):
        super(Pi, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 256),nn.PReLU(),nn.Dropout(),
            nn.Linear(256, 128), nn.PReLU(),nn.Dropout(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        action_val = self.fc(x)
        m = nn.Softmax(dim=1)
        action_prob = m(action_val)
        return action_prob

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
    env = gym.make("CartPole-v1")

    if render_flag:
        env.render()

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
                state = torch.from_numpy(state).unsqueeze(0).to(device)
                action_prob = pi(state)

                # choose the better action and get log of action prob
                m = dist.Categorical(action_prob)
                action = m.sample().to(device)
                log_action_prob = m.log_prob(action).to(device)
                state, reward, episode_flag, _ = env.step(action.item())


                # save rollouts
                cur_reward_list.append(reward)
                cur_log_prob_list.append(log_action_prob)

            total_reward += sum(cur_reward_list)
            reward_list.append(cur_reward_list)
            log_prob_list.append(cur_log_prob_list)
            #end of episode

        return_list = get_return(reward_list)
        cur_policy_loss = calculate_loss(return_list, log_prob_list)

        #save avg reward/episode
        avg_reward_list.append(total_reward/maxepisode)

        print("Iter", i, "avg reward", avg_reward_list[-1])

        # policy_gradient_update
        optimizer.zero_grad()
        cur_policy_loss.backward()
        optimizer.step()
        #end of iteration

    #save result
    output = np.array(avg_reward_list)
    np.save("./part_" + str(part) + "_ep_" + str(maxepisode), output)
    print("End: part " + str(part))

def main():
    global part
    global maxepisode

    for i in range(1,5):
        part = i
        print("Start: part " + str(part))

        if i == 4:
            episode_lists = [100, 300, 1000]
            for ep in episode_lists:
                maxepisode = ep
                print(maxepisode)
                execute()
        else:
            execute()

if __name__ == '__main__':
    main()