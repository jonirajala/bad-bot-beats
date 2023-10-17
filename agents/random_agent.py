from pypokerengine.players import BasePokerPlayer
import random

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
