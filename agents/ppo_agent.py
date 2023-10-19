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




NB_SIMULATION = 1000

class PPOPlayer(BasePokerPlayer):  # Do not forget to make parent class as "BasePokerPlayer"
    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state, policy_net=None):
        if policy_net:
            community_card = np.zeros(5)
            community_card[:len(round_state['community_card'])] = [card.id for card in gen_cards(round_state['community_card'])]
            hole_card = np.array([card.to_id() for card in gen_cards(hole_card)])
            pot = round_state['pot']['main']['amount']
            street = street_to_id(round_state['street'])
            big_blind_pos = round_state['big_blind_pos']

            pot_array = np.array([pot])
            street_array = np.array([street])
            big_blind_pos_array = np.array([big_blind_pos])

            state = torch.tensor(np.concatenate((community_card, hole_card, pot_array, street_array, big_blind_pos_array)))

            probs = policy_net(state)
            action = np.random.choice(len(probs[0]), p=probs[0].detach().numpy())
        else:
            action = random.randint(0,3)

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
        # player_num = game_info["player_num"]
        # max_round = game_info["rule"]["max_round"]
        # small_blind_amount = game_info["rule"]["small_blind_amount"]
        # ante_amount = game_info["rule"]["ante"]
        # blind_structure = game_info["rule"]["blind_structure"]
        
        # self.emulator = Emulator()
        # self.emulator.set_game_rule(player_num, max_round, small_blind_amount, ante_amount)
        # self.emulator.set_blind_structure(blind_structure)
        
        # # Register algorithm of each player which used in the simulation.
        # for player_info in game_info["seats"]["players"]:
        #     self.emulator.register_player(player_info["uuid"], SomePlayerModel())
        self.nb_player = game_info['player_num']

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
            nn.Linear(INPUT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, OUPUT_DIM),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

class ValueNetwork(nn.Module):
    def __init__(self, INPUT_DIM):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(INPUT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.fc(x)
