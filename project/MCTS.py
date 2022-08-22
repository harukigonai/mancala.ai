import numpy as np
from math import sqrt


class MCTS:
    def __init__(self, c_puct, num_actions):
        self.c_puct = c_puct
        self.num_actions = num_actions
        self.visited = set()
        self.Q = dict()
        self.N = dict()
        self.P = dict()

    def calculate_ucb(self, s, a):
        return self.Q[s][a] + self.c_puct * self.P[s][a] * \
               sqrt(sum(self.N[s])) / (1 + self.N[s][a])

    def update_Q(self, s, a, v):
        self.Q[s][a] = (self.N[s][a] * self.Q[s][a] + v) / (self.N[s][a] + 1)

    def search(self, current_s, nnet):
        # Select
        # Keep taking actions to states with highest UCB value
        v = None
        s = current_s
        s_a_li = []
        while True:
            # Expand and Evaluate
            if s.is_terminal():
                v = -s.game_reward()
                break
            elif s not in self.visited:
                self.P[s], v = nnet.predict(s)
                self.Q[s] = np.zeros(self.num_actions)
                self.N[s] = np.zeros(self.num_actions)
                self.visited.add(s)
                v *= -1
                break

            max_u = -float("inf")
            best_a = -1
            for a in s.get_valid_actions():
                u = self.calculate_ucb(s, a)
                if u > max_u:
                    max_u = u
                    best_a = a
            a = best_a
            s_a_li.append((s, a))

            s = s.next_state(a)

        # Backup
        # Update Q and N values
        for s_a_tuple in s_a_li:
            s, a = s_a_tuple
            self.update_Q(s, a, v)
            self.N[s][a] += 1
