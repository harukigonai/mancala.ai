from MCTS import MCTS
import numpy as np
from multiprocessing import Pool, set_start_method


class Learner:
    def __init__(self, num_nns, num_eps, num_MCTS_sims,
                 c_puct, pit_num_games, pit_win_threshold):
        self.num_nns = num_nns
        self.num_eps = num_eps
        self.num_MCTS_sims = num_MCTS_sims
        self.c_puct = c_puct
        self.pit_num_games = pit_num_games
        self.pit_win_threshold = pit_win_threshold

    def execute_episodes(self, game_class, nn_class):
        # data = []

        # print("- Starting episodes")
        # threads = [None] * self.num_eps
        # for i in range(len(threads)):
        #     threads[i] = Thread(target=self.execute_episode,
        #                         args=(game_class, nn, data))
        #     threads[i].start()

        # print("- Waiting for episodes")
        # for i in range(len(threads)):
        #     threads[i].join()

        # return data

        p = Pool()
        res = p.starmap(self.execute_episode,
                        [(game_class, nn_class)] * self.num_eps)
        p.close()
        return sum(res, [])

    def execute_episode(self, game_class, nn_class):
        nn = nn_class("curr")

        data = []
        s = game_class.start_state()
        mcts = MCTS(self.c_puct, game_class.get_num_actions())

        while True:
            for i in range(self.num_MCTS_sims):
                mcts.search(s, nn)
            state_info = {
                's': s,
                'P': mcts.P[s],
                'v': None
            }
            data.append(state_info)

            P = np.multiply(mcts.P[s], s.get_valid_actions_mask())
            P = P / sum(P)
            a = np.random.choice(len(P), p=P)

            print("predicting", s.board, s.pit_pos_1, s.pit_neg_1, s.turn, a)

            s = s.next_state(a)

            if s.is_terminal():
                break
        self.assign_rewards(data, s)

        print("- Episode terminated")
        return data

    def assign_rewards(self, data, term_s):
        for state_info in data:
            state = state_info['s']
            state_info['v'] = state.calculate_state_reward(term_s)

    def nn_get_action(self, nn, game, s):
        pi, _ = nn.predict(s)
        P = np.multiply(pi, game.get_valid_actions_mask(s))
        P = P / sum(P)
        return np.random.choice(len(P), p=P)

    def pit_nns(self, game, nn_class):
        # winner_li = []

        # print("- Starting pitting threads")
        # threads = [None] * self.pit_num_games
        # for i in range(len(threads)):
        #     threads[i] = Thread(target=self.pit_nns_once,
        #                         args=(game, nn_1, nn_2, winner_li))
        #     threads[i].start()

        # print("- Waiting for pitting threads")
        # for i in range(len(threads)):
        #     threads[i].join()

        # nn_1_wins = winner_li.count(1)
        # return nn_1_wins / self.pit_num_games

        p = Pool()
        winner_li = p.starmap(self.pit_nns_once,
                              [(game, nn_class)] * self.pit_num_games)
        p.close()

        new_nn_wins = winner_li.count(1)
        return new_nn_wins / self.pit_num_games

    def pit_nns_once(self, game, nn_class):
        new_nn = nn_class("new")
        curr_nn = nn_class("curr")

        s = game.start_state()
        while not game.is_terminal(s):
            if s.turn == 1:
                a = self.nn_get_action(new_nn, game, s)
            else:
                a = self.nn_get_action(curr_nn, game, s)
            s = game.next_state(s, a)
        print("- Winner of pitting is", game.game_reward(s))
        return game.game_reward(s)

    def optimize_nn(self, game_class, nn_class):
        nn = nn_class("perm")
        set_start_method("spawn")

        data = []
        for i in range(self.num_nns):
            nn.save_weights("curr")

            print("Iteration", i + 1, "of", self.num_nns)
            print("1. Generating data through MCTS")
            # for j in range(self.num_eps):
            #     print("- Executing episode", j + 1, "of", self.num_eps)
            #     data += self.execute_episode(game_class, nn)

            data = self.execute_episodes(game_class, nn_class)

            print("2. Training NN on data")
            new_nn = nn_class(train_data=data)
            new_nn.save_weights("new")

            print("3. Pitting curr NN and new NN")
            frac_win = self.pit_nns(game_class, nn_class)

            print("4. Frac win of new NN", frac_win)
            if frac_win > self.pit_win_threshold:
                nn = new_nn
        nn.save_model()
        nn.save_weights("perm")
        return nn
