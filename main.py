from Learner import Learner

from Mancala.Mancala import Mancala
from Mancala.MancalaNeuralNetwork import MancalaNeuralNetwork

from timeit import timeit


args = {
    'num_nns': 1,
    'num_eps': 10,
    'num_MCTS_sims': 10,
    'c_puct': 0.5,
    'pit_num_games': 52,
    'pit_win_threshold': 0.5
}


def run():
    learner = Learner(args['num_nns'], args['num_eps'], args['num_MCTS_sims'],
                      args['c_puct'], args['pit_num_games'],
                      args['pit_win_threshold'])
    learner.optimize_nn(Mancala, MancalaNeuralNetwork)


def main():
    print(timeit('run()', setup="from __main__ import run"))


if __name__ == "__main__":
    main()
