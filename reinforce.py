import torch
from torch import optim
from torch import nn
import gym
import numpy as np


class BanditEnv(gym.Env):
    '''
    Toy env to test your implementation
    The state is fixed (bandit setup)
    Action space: gym.spaces.Discrete(10)
    Note that the action takes integer values
    '''

    def __init__(self):
        self.action_space = gym.spaces.Discrete(10)
        self.observation_space = gym.spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32)
        self.state = np.array([0])

    def reset(self):
        return np.array([0])

    def step(self, action):
        assert int(action) in self.action_space

        done = True
        s = np.array([0])
        r = float(-(action - 7) ** 2)
        info = {}
        return s, r, done, info


class Reinforce:
    def __init__(self, policy, env, optimizer):
        self.policy = policy
        self.env = env
        self.optimizer = optimizer

    @staticmethod
    def compute_expected_cost(trajectory, gamma, baseline):
        """
        Compute the expected cost of this episode for gradient backprop
        DO NOT change its method signature
        :param trajectory: a list of 3-tuple of (reward: Float, policy_output_probs: torch.Tensor, action: Int)
        NOTE: policy_output_probs will have a grad_fn, i.e., it's able to backpropagate gradients from your computed cost
        :param gamma: gamma
        :param baseline: a simple running mean baseline to be subtracted from the total discounted returns
        :return: a 2-tuple of torch.tensor([cost]) of this episode that allows backprop and updated baseline
        """
        cost = 0
        rewards, probs, actions = list(zip(*trajectory))
        T = len(rewards)
        #Computing the G_t
        discounted_reward = 0
        G = []
        for t in reversed(range(T)):
            discounted_reward = rewards[t] + gamma*discounted_reward
            G.insert(0, discounted_reward)
        G = torch.FloatTensor(G)
        #Baseline nonsense that I'll do later.
        #p = 0.9
        #G = (G - baseline)/torch.std(G, unbiased = False)
        #baseline = p*baseline + (1-p)*torch.mean(G)
        #Final cost functions.
        cost = 0
        for t in range(T):
            cost = cost - G[t]*torch.log(probs[t][actions[t]])
        return cost, baseline
        
    def train(self, num_episodes, gamma):
        """
        train the policy using REINFORCE for specified number of episodes
        :param num_episodes: number of episodes to train for
        :param gamma: gamma
        :return: self
        """

        baseline = 0
        total_reward_per_episode = []
        trajectory_lengths = []
        running_average_reward = 0
        running_average_cost = 0
        for episode_i in range(num_episodes):
            self.optimizer.zero_grad()
            trajectory, trajectory_length, total_reward = self.generate_episode()
            loss, baseline = self.compute_expected_cost(trajectory, gamma, baseline)
            loss.backward()
            self.optimizer.step()
            total_reward_per_episode.append(total_reward)
            print(episode_i)
            if episode_i%200 == 0 and episode_i!=0:
                running_average_reward = np.sum(total_reward_per_episode[episode_i - 200:episode_i])/episode_i
                print("Episode: %d Reward: %5d " % (episode_i, total_reward))
        #iterate over episodes to get costs and use gradient descent to minimize them. Also print out the progress. Need to figure out if you're calculating the right rewards/costs. 
            if episode_i%200 == 0 and episode_i!=0:
                torch.save(self.policy.state_dict(), 'mypolicy.pth')
                #print("Checkpoint created at Episode %d" % (episode_i))

        return self

    def generate_episode(self):
        """
        run the environment for 1 episode
        NOTE: do not limit the number
        :return: whatever you need for training
        """

        ### YOUR CODE HERE AND REMOVE `pass` below ###
        state = torch.FloatTensor(self.env.reset())
        total_reward = 0
        trajectory = []
        while True:
            probs = self.policy.forward(state)
            action = torch.distributions.Categorical(probs).sample()
            state, reward, finished, __ = env.step(action.item())
            total_reward = total_reward + reward
            state = torch.FloatTensor(state)
            trajectory.append([reward, probs, action.item()])
            if finished:
                break
            #if len(trajectory) == 5:
            #    break
        trajectory_length = len(trajectory)

        return trajectory, trajectory_length, total_reward



# Do NOT change the name of the class.
# This class should contain your policy model architecture.
# Please make sure we can load your model with:
# policy = MyPolicy()
# policy.load_state_dict(torch.load("mypolicy.pth"))
# This means you must give default values to all parameters you may wish to set, such as output size.
class MyPolicy(nn.Module):
    def __init__(self):
        super(MyPolicy, self).__init__()
        self.net_stack = nn.Sequential(
            nn.Linear(8, 25), 
            nn.ReLU(), 
            nn.Linear(25, 16), 
            nn.ReLU(), 
            nn.Linear(16, 16), 
            nn.ReLU(), 
            nn.Linear(16, 4), 
            nn.Softmax(dim = 0),
        )
        def kaim_init(m):
            if type(m) == nn.Linear:
                torch.nn.init.kaiming_uniform_(m.weight)
        self.net_stack.apply(kaim_init)



    def forward(self, x):
        ### YOUR CODE HERE AND REMOVE `pass` below ###
        result = self.net_stack(x)
        return result


if __name__ == '__main__':
    policy = MyPolicy()
    optimizer = optim.Adam(policy.parameters(), lr = 0.0001) 
    env = gym.make('LunarLander-v2')
    learner = Reinforce(policy, env, optimizer)
    learner.train(10000, 0.99)
    torch.save(model.state_dict(), 'mypolicy.pth')