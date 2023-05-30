"""REINFORCE algorithm.

Author: Elie KADOCHE.
"""

import torch
from torch import optim as optim
from torch.distributions import Categorical

from src.envs.cartpole import CartpoleEnv
from src.models.actor import ActorModel
from collections import deque

# Policy model path
ACTOR_PATH = "models/actor.pt"

# Maximum environment length
HORIZON = 500

# ---> TODO: change the discount factor to solve the problem

#We tested 0.90 and 0.99. Both got similar results
DISCOUNT_FACTOR = 0.90

# ---> TODO: change the learning rate to solve the problem

# we tested 0.1, 0.01, and 0.001
# 0.1 did not converge
# 0.01 was unstable
# 0.001 proved the best of the three tests in terms of speed and stability
LEARNING_RATE = 0.001

#training will stop when we achieve the maximum score STOPAGE amount of times
STOPAGE = 10

# cirular queue will store the last STOPAGE rewards
final_reward_queue = deque(maxlen = STOPAGE)

if __name__ == "__main__":
    # Create policy
    actor = ActorModel()
    actor.train()

    # Create the environment
    env = CartpoleEnv()

    # Create optimizer with the policy parameters
    actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)

    # ---> TODO: when do we stop the training?

    # Run infinitely many episodes
    training_iteration = 0
    while True:

        # Experience
        # ------------------------------------------

        # Reset the environment
        state, _ = env.reset()

        # During experience, we will save:
        # - the probability of the chosen action at each time step pi(at|st)
        # - the rewards received at each time step ri
        saved_probabilities = list()
        saved_rewards = list()

        # Prevent infinite loop
        for t in range(HORIZON + 1):

            # Use the policy to generate the probabilities of each action
            probabilities = actor(state)

            # Create a categorical distribution over the list of probabilities
            # of actions and sample an action from it
            distribution = Categorical(probabilities)
            action = distribution.sample()
            # Take the action
            state, reward, terminated, _, _ = env.step(action.item())

            # Save the probability of the chosen action and the reward
            saved_probabilities.append(probabilities[0][action])
            saved_rewards.append(reward)

            # End episode
            if terminated:
                break

        # Compute discounted sum of rewards
        # ------------------------------------------

        # Current discounted reward
        discounted_reward = 0.0

        # List of all the discounted rewards, for each time step
        discounted_rewards = list()

        # ---> TODO: compute discounted rewards
        # new_r represents the cumulative discounted reward at each time step.
        new_r = 0
        # we work backwards to efficiently calculate the powers of DISCOUNT_FACTOR
        for r in saved_rewards[::-1]:
            # the DISCOUNT_FACTOR's exponent increases by one for every future reward
            new_r = r + DISCOUNT_FACTOR*new_r
            # append it to the list
            discounted_rewards.append(new_r)
        #restore temporal order
        discounted_rewards.reverse()


        # Eventually normalize for stability purposes
        discounted_rewards = torch.tensor(discounted_rewards)
        mean, std = discounted_rewards.mean(), discounted_rewards.std()
        discounted_rewards = (discounted_rewards - mean) / (std + 1e-7)

        # Update policy parameters
        # ------------------------------------------

        # For each time step
        actor_loss = list()
        for p, g in zip(saved_probabilities, discounted_rewards):
            # ---> TODO: compute policy loss

            # loss = discounted reward * log of the policy probability
            # we put the negative so that backpropagation peroforms gradient ASCENT
            time_step_actor_loss = -g * torch.log(p) 

            # Save it
            actor_loss.append(time_step_actor_loss)

        # Sum all the time step losses
        actor_loss = torch.cat(actor_loss).sum()

        # Reset gradients to 0.0
        actor_optimizer.zero_grad()

        # Compute the gradients of the loss (backpropagation)
        actor_loss.backward()

        # Update the policy parameters (gradient ascent)
        actor_optimizer.step()

        # Logging
        # ------------------------------------------

        # Episode total reward
        episode_total_reward = sum(saved_rewards)

        # ---> TODO: when do we stop the training?

        # Log results
        log_frequency = 5
        training_iteration += 1
        if training_iteration % log_frequency == 0:

            # Save neural network
            torch.save(actor, ACTOR_PATH)

            # Print results
            print("iteration {} - last reward: {:.2f}".format(
                training_iteration, episode_total_reward))

        # ---> TODO: when do we stop the training?
        # stop training when maximum reward is achieved STOPAGE consecutive times
        final_reward_queue.append(int(episode_total_reward))
        # we check it by comparing the sum of the circular queue
        if sum(final_reward_queue) == STOPAGE*HORIZON:
            break

