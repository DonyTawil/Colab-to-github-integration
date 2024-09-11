
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:                                # logging
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),                    # not sure why Tanh (I thought neural networks should be only positive so that the output is +, does this mean output can be negative? wtf )
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Tanh()
                        )
        else:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )
        # critic   The value network lmao, original comment was just critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)  # Create a tensor with value new_action_std * new_action_std and with specified dimension
        else:                                                                                             # This I believe is the variance of the action, with the action being made of a probability distribution (mean, variance)
            print("--------------------------------------------------------------------------------------------")     # the mean will be added later if I understood it correctly
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)         # Get the action mean from the neural network that takes the state as input
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0) # Extract the diagonal matrix (why not start with a diag from the start), unsqueeze to nest the tensor adding one dimension,
            print("cov_mat: ")
            print(cov_mat)                                                       # make it compatible with the rest of the code
            dist = MultivariateNormal(action_mean, cov_mat)        # Generate the probability for the action, note that the probabilites are also continuous (This also mean that only actions
                                                                   # that are neach other are similarly probable, would it be a good idea then to have a probability that is not gaussian?
                                                                   # i.e allowing actions far away from each other to be local extremums, in the explanation below for example action = 0.5 would have probaility 0.3 and also action 0.75 = 0.3 as well for example)
                                                                   # More explanation: let the action space be the closed continuous set [0, 1], the mean being 0.5 with probability 0.3
                                                                   # for action 0.49 the probability will be close but somewhat less than 0.3
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()                                   # Take a random action, while respecting the probabilities
                                                                 # (actions and states are tensors as well (remember) and their dimensions are given before)
        action_logprob = dist.log_prob(action)                   ### log_prob calculated to help update the PPO algorithm later, is this overwritten or used later
        state_val = self.critic(state)                           # Guess the reward using the value network

        return action.detach(), action_logprob.detach(), state_val.detach()   # Detach means stop tracking the values in the computational graph, this tracking to make backpropagation easier (as explained by chatgpt lol)

    def evaluate(self, state, action):
        # Dunno all the details, but used to evaluate and update the PPO algo, read into it later

        if self.has_continuous_action_space:
            action_mean = self.actor(state)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):

        # Small half reminder on how this thing works, we have the surrogate function after a training iteration
        # we have the old policy, and other functions, we write the equations in terms of the new policy
        # (note for action at in the surrogate objective function we only look at weather or not it was advantageous to take action at
        # and if so does the new policy take or not this action)
        # we then use stochastic gradient descent (I believe with backpropagation) to then calculate the new policy (the new parameters) that maximizes the reward

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma             # Discount factor, to discount future rewards (if between 0 and 1), values between [0, 1] means future rewards are less important than immediate rewards
        self.eps_clip = eps_clip       # clipping of the change in PPO policy (remind equations and stuff) (1 + epsi, 1 - epsi)  is the max and min changes allowed
        self.K_epochs = K_epochs       # Number of epochs for training of the new ppo policy

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()       # Initialize the mean square error function

    def set_action_std(self, new_action_std):                 # Set the new action standard, surprised that policy_old get the new standard (deviation) as well
        if self.has_continuous_action_space:                  # chatgpt told me it is to have fair and accurate representation and also if we don't do this we might have unusable
            self.action_std = new_action_std                  # probability, I think it is mainly to avoid an unusable probability
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
       # Reduce action variance, this has the goal of reducing the exploration and increasing exploitation
       # there is a min variance, also I don't see this function used in other class functions? Check out where used in other files
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():                                                # no_grad: context manager to not compute gradient, chatgpt: sets all the requires_grad flags to False.
                state = torch.FloatTensor(state).to(device)                      # This is used to disable gradient calculation. gradient is calculated with loss.backward()
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)                                    # logging
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob, state_val = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):    # buffer.rewards is filled through the environment, start in the reverse
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)    # Explaining the code because it is somewhat implicit
            rewards.insert(0, discounted_reward)                             # Starting with last obtained reward (discount is 0 here), than the last obtained + last obtained reward * gamma
                                                                             # And then storing it in rewards, so rewards stores all the rewards at each step
                                                                             # (counting rewards from previous step) * discount_factor :P
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()            # Actual rewards - guessed rewards using the value network
                                                                             # #### IMP ############ which was more recently trained on the old policy i.e the old old policy not this old policy fuck me! because the advantage function is actually for the previous itération per se i.e for the old new policy)
                                                                             # Now I am confused about the ratios again. I am confused because advantages use the old policy only whereas the ratio contains both
                                                                             # Anyway after asking perplexity I think I understand now, logprobs I believe give the probability of choosing action a, so using the advantage functions
                                                                             # We increase the probability of taking action a or reducing it based on what is more useful, next iteration if a for example is disadvantageous, next iteration (After reducing probability of a enough)
        # Optimize policy for K epochs                                       # we might take a new action and therefore evaluate (for this pass) this action
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)  # Another important point, the new policy at first is random
                                                                                                  # and is used to get the equation for the loss function
            # match state_values tensor dimensions with rewards tensor                            # with the loss function we do gradient descent and that allow us to calculate
            state_values = torch.squeeze(state_values)                                            # the new weights of the policy. ::: This surprised me I felt more that the new policy would be
                                                                                                  # close to the old policy with some small shifts in weights, but I m not sure doing it like this
            # Finding the ratio (pi_theta / pi_theta__old)                                        # would give any advantage
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy     # A reminder about how this works, goal is to minimze loss function
                                                                                                                  # i.e get loss to the minimal value possible even if below zero
                                                                                                                  # - dist_entropy (even though it is a numerical value and not an equation to minimise)
                                                                                                                  # helps us make sure to explore more (by maximizing entropy which is done here by implicitly because entropy is not an equation)
                                                                                                                  # Finally MSE.error so that our value network is accurate so here we reduce the error
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())  # Because old_policy is now the current policy

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


