from pypokerengine.api.game import setup_config, start_poker
from agents.random_agent import RandomPlayer
from agents.calling_machine_agent import CallingMachinePlayer
from agents.probability_agent import ProbabilityPlayer
from agents.ppo_agent import PPOPlayer, PolicyNetwork, ValueNetwork
import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import get_action_by_num
from utils import state_to_tensor
from pypokerengine.api.emulator import Const
from emulators.emulator import MyEmulator
import numpy as np

ITERS = 20
INITIAL_STACK = 200
SB = 1

# PPO Constants
CLIP_EPSILON = 0.2
EPOCHS = 10
INPUT_DIM = 10
OUPUT_DIM = 4

# def simulate():
#     winner = {"random_player":0, "ai_player":0, "tie":0}
#     money = {"random_player":0, "ai_player":0}

#     for iter in range(ITERS):
#         config = setup_config(max_round=100, initial_stack=INITIAL_STACK, small_blind_amount=SB)
#         config.register_player(name="random_player", algorithm=PPOAgent())
#         config.register_player(name="ai_player", algorithm=ProbabilityPlayer())
#         game_result = start_poker(config, verbose=0)

#         players = game_result['players']
#         if players[0]['stack'] > players[1]['stack']:
#             winner[players[0]['name']] += 1
#         elif players[0]['stack'] < players[1]['stack']:
#             winner[players[1]['name']] += 1
#         else:
#             winner['tie'] += 1
        
#         money[players[0]['name']] += players[0]['stack'] - 200
#         money[players[1]['name']] += players[1]['stack'] - 200

#         # if iter % 100 == 0:
#         #     print(f"game: {iter}")
#         print(iter)

#     print(f"max hands played: {ITERS * 100}")
#     print(winner)
#     print(money)

# if __name__ == "__main__":
#     simulate()




# PPO Update
def ppo_update(states, actions, returns, old_probs, policy_net, value_net, policy_optimizer, value_optimizer, clip_epsilon=CLIP_EPSILON):
    EPOCHS = 10
    states = torch.stack(states, 0)
    actions = torch.tensor(actions).reshape(-1,1)
    old_probs = torch.tensor(old_probs)
    returns = torch.tensor(returns).float()

    value_losses = []
    policy_losses = []
    
    for _ in range(EPOCHS):
        # Compute current action probabilities

        current_probs = policy_net(states).gather(1, actions)
        ratio = current_probs / old_probs

        # Compute values
        values = value_net(states).squeeze(1)
        advantage = returns - values.detach()

        # Compute surrogate objective
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()

        # Update policy
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        # Update value function
        value_loss = nn.MSELoss()(returns, values)
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()

        value_losses.append(value_loss.detach())
        policy_losses.append(policy_loss.detach())

    return value_losses, policy_losses, policy_net, value_net

def collect_trajectories(env, policy_net, value_net, num_episodes=100, gamma=0.99):
    states, actions, rewards, old_probs, returns = [], [], [], [], []
    players_info = {
        "1": { "name": "random", "stack": 100 },
        "2": { "name": "ppo", "stack": 100 },
    }
    for episode in range(num_episodes):
        initial_state = env.generate_initial_game_state(players_info)
        episode_rewards = []
        episode_values = []
        episode_states = []
        episode_actions = []
        episode_probs = []
        wons = 0
        
        msgs = []
        
        game_state, events = env.start_new_round(initial_state)

        episode_reward = 0        
        # while game_state["street"] != Const.Street.FINISHED:
        while True:
            a = env.run_until_my_next_action(game_state, "2", msgs)
            if len(a) == 4:
                game_state, valid_actions, hole_card, round_state = a
                
                X = state_to_tensor(round_state, hole_card).float()
                
                action_probs = policy_net(X).detach().numpy()
                action_num = np.random.choice(len(action_probs), p=action_probs)

                action, amount = get_action_by_num(action_num, valid_actions)
                # print(action)
                game_state, msgs = env.apply_my_action(game_state, action, amount)

                # Store values
                episode_states.append(X)
                episode_actions.append(action_num)
                episode_rewards.append(0)
                episode_probs.append(action_probs[action_num])
                # episode_values.append(value_net(X).item())
            else:
                game_state, reward = a       
                episode_reward = reward
                # print(action)
                # print(episode_reward)
                # if episode_reward < 0:
                #     print("eslkdfjklsefmkldsm")
                wons += episode_reward
                break
        

        # print(round_state)
        # Compute returns for the episode
        episode_returns = []
        G = 0
        # for r in reversed(episode_rewards[1:]):
        for i in range(len(episode_states)):
            if i == 0:
                G = episode_reward
            else:
                G = gamma * G
            episode_returns.insert(0, G)
        
        # Compute advantages
        # episode_advantages = [G - v for G, v in zip(episode_returns, episode_values)]

        # Append episode values to global lists
        states.extend(episode_states)
        actions.extend(episode_actions)
        rewards.extend(episode_rewards)
        old_probs.extend(episode_probs)
        returns.extend(episode_returns)
        # advantages.extend(episode_advantages)

    return states, actions, rewards, old_probs, returns, wons


def train():
    # Initialize networks and optimizers
    
    policy_net = PolicyNetwork(INPUT_DIM, OUPUT_DIM)
    value_net = ValueNetwork(INPUT_DIM)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    value_optimizer = optim.Adam(value_net.parameters(), lr=0.001)

    env = MyEmulator()
    env.set_game_rule(player_num=2, max_round=100, small_blind_amount=1, ante_amount=0)
    env.register_player(uuid="2", player=PPOPlayer())
    env.register_player(uuid="1", player=ProbabilityPlayer())
    

    # while True:
    #     states, actions, rewards, old_probs, returns, advantages = collect_trajectories(env, policy_net, value_net, num_episodes=100, gamma=0.99)
    #     ppo_update(states, actions, returns, old_probs)
    value_losses, policy_losses, all_returns = [], [], [0]
    for i in range(ITERS):
        print(f"Round {i+1}/{ITERS}")
        states, actions, rewards, old_probs, returns, wons = collect_trajectories(env, policy_net, value_net, num_episodes=100, gamma=0.99)
        value_loss, policy_loss, policy_net, value_net = ppo_update(states, actions, returns, old_probs, policy_net, value_net, policy_optimizer, value_optimizer)
        value_losses += value_loss
        policy_losses += policy_loss
        all_returns.append(wons+all_returns[-1])
    
    fig, ax = plt.subplots(ncols=3, figsize=(10,4))


    ax[0].plot(value_losses)
    ax[1].plot(policy_losses)
    ax[2].plot(all_returns)

    ax[0].set_title("Value losses")
    ax[1].set_title("Policy losses")
    ax[2].set_title("Bank roll")

    plt.show()

if __name__ == "__main__":
    train()