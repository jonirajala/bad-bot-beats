import torch
import numpy as np
from pypokerengine.utils.card_utils import gen_cards

def card_to_id(card_str):
    # Define mappings for suits and ranks
    suit_mapping = {'H': 1,  # Hearts
                    'D': 2,  # Diamonds
                    'C': 3,  # Clubs
                    'S': 4}  # Spades
    
    rank_mapping = {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, 
                    '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 
                    'J': 11, 'Q': 12, 'K': 13}
    
    # Split card string into suit and rank
    suit = card_str[0]
    rank = card_str[1:]
    
    # Convert to unique ID
    # We multiply the suit by 100 to ensure a unique ID for each card
    card_id = suit_mapping[suit] * 100 + rank_mapping[rank]
    
    return card_id

def street_to_id(street_str):
    street_mapping = {'preflop': 0,
                      'flop': 1,
                      'turn':2,
                      'river':3}
    return street_mapping[street_str]


def get_action_by_num(action, valid_actions):
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


def state_to_tensor(round_state, hole_card, device):
    community_card = np.zeros(5)
    community_card[:len(round_state['community_card'])] = [card.to_id() for card in gen_cards(round_state['community_card'])]
    hole_card = np.array([card.to_id() for card in gen_cards(hole_card)])
    pot = round_state['pot']['main']['amount']
    street = street_to_id(round_state['street'])
    big_blind_pos = round_state['big_blind_pos']
    # Convert single numbers to 1D numpy arrays with 1 element for concatenation
    pot_array = np.array([pot])
    street_array = np.array([street])
    big_blind_pos_array = np.array([big_blind_pos])

    state = torch.tensor(np.concatenate((community_card, hole_card, pot_array, street_array, big_blind_pos_array)), device=device, dtype=torch.float32)
    return state