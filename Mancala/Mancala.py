import numpy as np
import random
import copy

from Game import Game

P_POS_1_PIT = 12
P_NEG_1_PIT = 13


class Mancala(Game):
    def __init__(self, board, pit_pos_1, pit_neg_1, turn):
        self.board = board
        self.pit_pos_1 = pit_pos_1
        self.pit_neg_1 = pit_neg_1
        self.turn = turn

    # Initialize and return a random start state
    @classmethod
    def start_state(cls):
        board = np.zeros(12)
        for i in range(0, 6):
            board[i] = random.randint(1, 5)
        for i in range(6, 12):
            board[i] = random.randint(1, 5)

        return Mancala(board, 0, 0, 1)

    @classmethod
    def get_num_actions(cls):
        return 12

    # Given state and action, return next state
    def next_state(self, a):
        s_cpy = copy.deepcopy(self)

        move_is_over = False
        pebbles_in_hand = s_cpy.board[a]
        s_cpy.board[a] = 0

        curr_sp = (a + 1) % 12

        last_pit = -1
        while not move_is_over:
            while pebbles_in_hand != 0:
                # Put the pebble down
                s_cpy.board[curr_sp] += 1
                pebbles_in_hand -= 1
                last_pit = curr_sp

                if pebbles_in_hand != 0:
                    if curr_sp == 5 and s_cpy.turn == 1:
                        s_cpy.pit_pos_1 += 1
                        pebbles_in_hand -= 1
                        last_pit = P_POS_1_PIT
                    elif curr_sp == 11 and s_cpy.turn == -1:
                        s_cpy.pit_neg_1 += 1
                        pebbles_in_hand -= 1
                        last_pit = P_NEG_1_PIT
                curr_sp = (curr_sp + 1) % 12

            if (last_pit != P_POS_1_PIT and last_pit != P_NEG_1_PIT) \
               and s_cpy.board[last_pit] != 1:

                pebbles_in_hand = s_cpy.board[last_pit]
                s_cpy.board[last_pit] = 0
            else:
                move_is_over = True

        if not (last_pit == P_POS_1_PIT or last_pit == P_NEG_1_PIT):
            s_cpy.turn = -s_cpy.turn

        return s_cpy

    def get_valid_actions(self):
        valid_actions = set()
        if self.turn == 1:
            for i in range(0, 6):
                if self.board[i] != 0:
                    valid_actions.add(i)
        elif self.turn == -1:
            for i in range(6, 12):
                if self.board[i] != 0:
                    valid_actions.add(i)
        return valid_actions

    def get_valid_actions_mask(self):
        valid_actions = np.zeros(12)
        if self.turn == 1:
            for i in range(0, 6):
                if self.board[i] != 0:
                    valid_actions[i] = 1
        elif self.turn == -1:
            for i in range(6, 12):
                if self.board[i] != 0:
                    valid_actions[i] = 1
        return valid_actions

    def game_reward(self):
        return 1 if self.pit_pos_1 > self.pit_neg_1 else -1

    def is_terminal(self):
        return sum(self.board[0:6]) == 0 or sum(self.board[6:12]) == 0

    def calculate_state_reward(self, term_s):
        reward = term_s.game_reward()
        return self.turn * reward
