from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate

NB_SIMULATION = 1000

class ProbabilityPlayer(BasePokerPlayer):  # Do not forget to make parent class as "BasePokerPlayer"
    #  we define the logic to make an action through this method. (so this method would be the core of your AI)
    def declare_action(self, valid_actions, hole_card, round_state):
        community_card = round_state['community_card']
        
        win_rate = estimate_hole_card_win_rate(
                nb_simulation=NB_SIMULATION,
                nb_player=2,
                hole_card=gen_cards(hole_card),
                community_card=gen_cards(community_card)
                )
        
        exp_value = win_rate * (round_state['pot']['main']['amount']+valid_actions[1]['amount'])-valid_actions[1]['amount']
        # print(round_state['pot']['main']['amount'], win_rate, exp_value)

        if exp_value >= 0:
            # some how implement bettting also
            if win_rate >= 1/2:
                action = valid_actions[2]  # fetch CALL action info
                amount = int((action['amount']['min'] + action['amount']['max']) / 4)
                action = action['action']
            else:    
                action = valid_actions[1]  # fetch CALL action info
                amount = action['amount']
                action = action['action']
                
        else:
            action = valid_actions[0]  # fetch CALL action info
            amount = action['amount']
            action = action['action']

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
    return ProbabilityPlayer()



