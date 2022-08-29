from xml.dom import ValidationErr
from Learner import Learner

from Mancala.Mancala import Mancala
from Mancala.MancalaNeuralNetwork import MancalaNeuralNetwork

from timeit import timeit

import sys

# args = {
#     'num_nns': 1,
#     'num_eps': 10,
#     'num_MCTS_sims': 10,
#     'c_puct': 0.5,
#     'pit_num_games': 52,
#     'pit_win_threshold': 0.5
# }


def run():
    if len(sys.argv) != 7:
        err = "Format: python3 main.py [num_nns] [num_eps] " + \
              "[num_MCTS_sims] [c_puct] [pit_num_games] [pit_win_threshold]"
        raise ValidationErr(err)

    args = sys.argv
    num_nns = int(args[1])
    num_eps = int(args[2])
    num_MCTS_sims = int(args[3])
    c_puct = float(args[4])
    pit_num_games = int(args[5])
    pit_win_threshold = float(args[6])

    learner = Learner(num_nns, num_eps, num_MCTS_sims,
                      c_puct, pit_num_games,
                      pit_win_threshold)
    learner.optimize_nn(Mancala, MancalaNeuralNetwork)


def main():
    time_elapsed = timeit('run()', number=1, setup="from __main__ import run")
    print("Time elapsed:", time_elapsed)


if __name__ == "__main__":
    main()
