from Learner import Learner

from Mancala.Mancala import Mancala
from Mancala.MancalaNeuralNetwork import MancalaNeuralNetwork

from timeit import timeit


args = {
    'num_nns': 1,
    'num_eps': 1,
    'num_MCTS_sims': 1,
    'c_puct': 0.5,
    'pit_num_games': 10,
    'pit_win_threshold': 0.5
}


def run():
    learner = Learner(args['num_nns'], args['num_eps'], args['num_MCTS_sims'],
                      args['c_puct'], args['pit_num_games'],
                      args['pit_win_threshold'])
    learner.optimize_nn(Mancala, MancalaNeuralNetwork)


def main():
    time_elapsed = timeit('run()', number=1, setup="from __main__ import run")
    print("Time elapsed:", time_elapsed)


if __name__ == "__main__":
    main()
