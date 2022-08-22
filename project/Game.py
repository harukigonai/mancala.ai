from abc import ABC, abstractmethod


class Game(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def start_state(cls):
        pass

    @abstractmethod
    def get_num_actions(cls):
        pass

    @abstractmethod
    def next_state(self, a):
        pass

    @abstractmethod
    def get_valid_actions(self):
        pass

    @abstractmethod
    def get_valid_actions_mask(self):
        pass

    @abstractmethod
    def game_reward(self):
        pass

    @abstractmethod
    def is_terminal(self):
        pass

    @abstractmethod
    def calculate_state_reward(self, term_s):
        pass
