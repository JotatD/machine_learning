"""REINFORCE (with baseline) algorithm.

Author: Elie KADOCHE.
"""

import torch
from torch import optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.nn as nn

import gym
import tqdm
from src.envs.cartpole import CartpoleEnv
from src.models.actor import ActorModel
from src.models.critic import CriticModel
from collections import deque
import numpy as np
# Policy and critic model path
ACTOR_PATH = "models/actor_bis.pt"
CRITIC_PATH = "models/critic.pt"

# Maximum environment length
HORIZON = 500

# discount factor to solve the problem
DISCOUNT_FACTOR = 0.99

# learning rate to solve the problem
LEARNING_RATE = 0.00075

#training will stop when we achieve the maximum score STOPAGE amount of times
STOPAGE = 10

# cirular queue will store the last STOPAGE rewards
final_reward_queue = deque(maxlen = STOPAGE)

# maximum number of episodes
EPOCHS = 10000

if __name__ == "__main__":
    #Make environment
    env = CartpoleEnv()

    #Init network
    actor = ActorModel()
    actor.train()
    critic = CriticModel()
    critic.train()

    #Init optimizer
    actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    critic_optimizer = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

    # ---> TODO: based on the REINFORCE script, create the actor-critic script
    for episode in range(EPOCHS):
        #init first stat
        state, _ = env.reset()
        #variable for end of the episode
        done = False
        # score of the running episode
        score = 0
        # loop discount factor, part of the loss funciton
        I = 1
        #run episode, update online

        #in this version of actor critic, we update our
        #policy in every step, not in every episode 
        for step in range(HORIZON+1):    

            #use network to predict action probabilities
            action_probs = actor(state)[0]
            
            #sample an action using the probability distribution
            m = Categorical(action_probs)
            action = m.sample()
            #get the log propability of that action
            lp = m.log_prob(action)

            #observe the next state, reward, and termination
            new_state, reward, done, _, _ = env.step(action.item())

            #update episode score
            score += reward
            
            #get state value of current state, according to our critic
            state_val = critic(state)
            
            #get state value of next state, according to our critic
            new_state_val = critic(new_state)
            
            #if terminal state, next state val is 0
            if done:
                new_state_val = torch.tensor([0]).float().unsqueeze(0)
            

            # First try, as stated in Sutton and Barto’s algorithm. Unsuccessful
            # advantage = reward + DISCOUNT_FACTOR * new_state_val.item() - state_val.item()
            # val_loss = - advantage * state_val
            ## Second try, multiplying loop discount (DISCOUNT_FACTOR^time_step). Unsuccessful
            # advantage = reward + DISCOUNT_FACTOR * new_state_val.item() - state_val.item()
            # val_loss = - advantage * state_val * I
            
            #Third try, letting the gradient step apply the chainrule for us
            #and multiplying by the loop discount. Successful
            val_loss = F.mse_loss(reward + DISCOUNT_FACTOR * new_state_val, state_val)
            val_loss *= I

            # from sutton and barto book. We use .item() because we dont want 
            #to backpropagte over the advantage
            advantage = reward + DISCOUNT_FACTOR * new_state_val.item() - state_val.item()          
            #calculate policy loss
            policy_loss = - lp * advantage * I
            
            #Backpropagate policy
            actor_optimizer.zero_grad()
            policy_loss.backward()
            ## for the first and second try we attempted with clipping the gradients
            # torch.nn.utils.clip_grad_norm_(actor.parameters(), 2.0)
            actor_optimizer.step()
            
            #Backpropagate value
            critic_optimizer.zero_grad()
            val_loss.backward()
            ## for the first and second try we attempted with clipping the gradients
            # torch.nn.utils.clip_grad_norm_(critic.parameters(), 2.0)
            critic_optimizer.step()
            
            if done:
                break
                
            #move into new state, discount I
            state = new_state
            I *= DISCOUNT_FACTOR
        
        if (episode % 5 == 0):
            print(f'episode {episode} - score {score}')
            # Save actor network
            torch.save(actor, ACTOR_PATH)
            # Save critc network
            torch.save(critic, CRITIC_PATH)
            
        # ---> TODO: when do we stop the training?
        # stop training when maximum reward achieved STOPAGE consecutive times
        final_reward_queue.append(int(score))
        # we check it by comparing the sum of the queue
        if sum(final_reward_queue) == STOPAGE*HORIZON:
            break
        