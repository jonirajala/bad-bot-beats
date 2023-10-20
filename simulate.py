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

# Training Constants
ITERS = 10
NUM_EPISODES = 200
EPOCHS = 10

# Poker Constants
INITIAL_STACK = 200
SB = 1

# PPO Constants
CLIP_EPSILON = 0.2
INPUT_DIM = 10
OUPUT_DIM = 4

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS activated")
else:
    device = torch.device("cpu")
    print ("MPS device not found.")


def ppo_update(states, actions, returns, old_probs, policy_net, value_net, policy_optimizer, value_optimizer, clip_epsilon=CLIP_EPSILON):
    EPOCHS = 10
    states = torch.stack(states, 0)
    actions = torch.tensor(actions, device=device).reshape(-1,1)
    old_probs = torch.tensor(old_probs, device=device)
    returns = torch.tensor(returns, device=device).float()

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

        value_losses.append(value_loss.cpu().detach())
        policy_losses.append(policy_loss.cpu().detach())

    return value_losses, policy_losses, policy_net, value_net

def collect_trajectories(env, policy_net, old_policy_net, num_episodes=100, gamma=0.99, debug=False):
    states, actions, rewards, old_probs, returns = [], [], [], [], []
    players_info = {
        "ppo_player": { "name": "ppo", "stack": INITIAL_STACK },
        "prob_opponent": { "name": "opponent", "stack": INITIAL_STACK },
    }
    action_counts = [0,0,0,0]
    wons = 0
    for episode in range(num_episodes):
        initial_state = env.generate_initial_game_state(players_info)
        episode_rewards = []
        episode_values = []
        episode_states = []
        episode_actions = []
        episode_probs = []
        
        
        msgs = []
        
        game_state, events = env.start_new_round(initial_state)

        episode_reward = 0        

        while True:
            a = env.run_until_my_next_action(game_state, "ppo_player", msgs, old_policy_net, device)
            if len(a) == 4:
                game_state, valid_actions, hole_card, round_state = a
                X = state_to_tensor(round_state, hole_card, device).float()
                
                action_probs = policy_net(X).cpu().detach().numpy()
                action_num = np.random.choice(len(action_probs), p=action_probs)

                action, amount = get_action_by_num(action_num, valid_actions)

                if debug:
                    print(action)
                    print(hole_card)
                    print(round_state['community_card'])

                # print(action)
                game_state, msgs = env.apply_my_action(game_state, action, amount)

                # Store values
                action_counts[action_num] += 1
                episode_states.append(X)
                episode_actions.append(action_num)
                episode_rewards.append(0)
                episode_probs.append(action_probs[action_num])
                # episode_values.append(value_net(X).item())
            else:
                game_state, end_stack = a       
                episode_reward = end_stack-INITIAL_STACK
                # print(end_stack)
                # print(episode_reward)
                # print(action)
                # print(episode_reward)
                
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

    return states, actions, rewards, old_probs, returns, wons, action_counts


def train():
    # Initialize networks and optimizers
    
    policy_net = PolicyNetwork(INPUT_DIM, OUPUT_DIM).to("mps")
    value_net = ValueNetwork(INPUT_DIM).to("mps")
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    value_optimizer = optim.Adam(value_net.parameters(), lr=0.001)

    env = MyEmulator()
    env.set_game_rule(player_num=2, max_round=100, small_blind_amount=1, ante_amount=0)
    env.register_player(uuid="ppo_player", player=PPOPlayer())
    # env.register_player(uuid="ppo_opponent", player=PPOPlayer())
    env.register_player(uuid="prob_opponent", player=ProbabilityPlayer())

    value_losses, policy_losses, all_returns, action_counts = [], [], [0], [0,0,0,0]

    debug = False
    old_policy_net = policy_net

    for i in range(ITERS):
        # if i > 30:
        #     debug = True
        print(f"Round {i+1}/{ITERS}")
        states, actions, rewards, old_probs, returns, wons, counts = collect_trajectories(env, policy_net, old_policy_net, num_episodes=NUM_EPISODES, gamma=0.99, debug=debug)
        
        old_policy_net = policy_net
        value_loss, policy_loss, policy_net, value_net = ppo_update(states, actions, returns, old_probs, policy_net, value_net, policy_optimizer, value_optimizer)

        if i > 0:
            for i in range(len(counts)):
                action_counts[i] += counts[i]
        value_losses += value_loss
        policy_losses += policy_loss
        all_returns.append(wons+all_returns[-1])

    
    print(f"Hands played: {NUM_EPISODES*ITERS}")
    print(f"Money won: {all_returns[-1]}")


    fig, ax = plt.subplots(ncols=4, figsize=(10,4))

    ax[0].plot(value_losses)
    ax[1].plot(policy_losses)
    ax[2].plot(all_returns)
    ax[3].bar([0,1,2,3],action_counts, tick_label=["Fold", "Call", "Small raise", "Big Raise"])

    ax[0].set_title("Value losses")
    ax[1].set_title("Policy losses")
    ax[2].set_title("Bank roll")
    ax[3].set_title("Action Counts")

    plt.show()

if __name__ == "__main__":
    train()