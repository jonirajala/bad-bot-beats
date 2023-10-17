from pypokerengine.api.game import setup_config, start_poker
from agents.random_agent import RandomPlayer
from agents.ai_agent import AIPlayer


ITERS = 1000


def simulate():
    winner = {"random_player":0, "ai_player":9, "tie":0}
    for iter in range(ITERS):
        config = setup_config(max_round=100, initial_stack=200, small_blind_amount=1)
        config.register_player(name="random_player", algorithm=RandomPlayer())
        config.register_player(name="ai_player", algorithm=AIPlayer())
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

if __name__ == "main":
    simulate()