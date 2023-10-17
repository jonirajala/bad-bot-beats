from pypokerengine.players import BasePokerPlayer
import random
from pypokerengine.api.game import setup_config, start_poker

class RandomPlayer(BasePokerPlayer):  # Do not forget to make parent class as "BasePokerPlayer"

    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state):
        action_info = random.choice(valid_actions)
        action, amount = action_info["action"], action_info["amount"]

        if action == "raise":
            amount = random.randint(valid_actions[2]['amount']['min'], valid_actions[2]['amount']['max'])
            return action, amount
        
        return action, amount   # action returned here is sent to the poker engine


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



ITERS = 1000

winner = {"random_player":0, "ai_player":9, "tie":0}

for iter in range(ITERS):
    config = setup_config(max_round=100, initial_stack=200, small_blind_amount=1)
    config.register_player(name="random_player", algorithm=RandomPlayer())
    config.register_player(name="ai_player", algorithm=RandomPlayer())
    game_result = start_poker(config, verbose=0)

    players = game_result['players']
    if players[0]['stack'] > players[1]['stack']:
        winner[players[0]['name']] += 1
    elif players[0]['stack'] < players[1]['stack']:
        winner[players[1]['name']] += 1
    else:
        winner['tie'] += 1
    
    if iter % 100 == 0:
        print(f"game: {iter}")

    
print(winner)
