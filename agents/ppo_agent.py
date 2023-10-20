from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards
import random
import numpy as np
import torch
from utils import street_to_id
from pypokerengine.players import BasePokerPlayer
import numpy as np
import torch
import torch.nn as nn
from utils import state_to_tensor



NB_SIMULATION = 1000

class PPOPlayer(BasePokerPlayer):  # Do not forget to make parent class as "BasePokerPlayer"
    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state, policy_net, device):
        X = state_to_tensor(round_state, hole_card, device).float()

        action_probs = policy_net(X).cpu().detach().numpy()
        action = np.random.choice(len(action_probs), p=action_probs)

        if action == 2: # small
            action_info = valid_actions[2]
            action = action_info["action"]
            amount = int((valid_actions[2]['amount']['min'] + valid_actions[2]['amount']['max']) / 4)
        
        elif action == 3: # big
            action_info = valid_actions[2]
            action = action_info["action"]
            amount = int((valid_actions[2]['amount']['min'] + valid_actions[2]['amount']['max'])*3 / 4)

        else:
            action_info = valid_actions[action]
            action, amount = action_info["action"], action_info["amount"]
        
        return action, amount


    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


def setup_ai():
    return PPOPlayer()


# Neural Network for Policy and Value Function
class PolicyNetwork(nn.Module):
    def __init__(self, INPUT_DIM, OUPUT_DIM):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(INPUT_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, OUPUT_DIM),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

class ValueNetwork(nn.Module):
    def __init__(self, INPUT_DIM):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(INPUT_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.fc(x)
