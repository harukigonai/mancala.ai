from Learner import Learner

from Mancala.Mancala import Mancala
from Mancala.MancalaNeuralNetwork import MancalaNeuralNetwork


def main():
    learner = Learner(1, 10, 10, 0.5, 52, 0.5)
    learner.optimize_nn(Mancala, MancalaNeuralNetwork)


if __name__ == "__main__":
    main()
