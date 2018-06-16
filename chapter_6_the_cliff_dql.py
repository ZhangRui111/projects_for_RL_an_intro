import numpy as np
import random

ALPHA = 0.2
EPSILON = 0.4
EPSILON_DECAY_LINE = 200000
GAMMA = 0.9
MAX_EPISODES = 1000000


class Cliff(object):
    """class Cliff:
    define the cliff environment:
        # -- Edge:any move to the edge will leave agent the original position.
        - -- Road:safe position with reward -1.
        * -- Cliff:unsafe position with reward -100 and then, put the agent at start point.
        S -- Start point.
        G -- Goal point with reward 100 and then, put the agent at start point.
        no.-- coordinate

          0 1 2 3 4 5 6 7
        0 # # # # # # # #
        1 # - - - - - - #
        2 # - - - - - - #
        3 # - - - - - - #
        4 # S * * * * G #
        5 # # # # # # # #

    define the action:
        0 -- up
        1 -- down
        2 -- left
        3 -- right
    """
    def __init__(self):
        self.current_p = [4, 1]
        self.reward = 0
        self.terminal = False

    def reset_cliff(self):
        self.reward = 0
        self.terminal = False
        """Reset the cliff, put the agent at start point.
        """
        self.current_p = [4, 1]
        return self.current_p

    def random_reset_cliff(self):
        self.reward = 0
        self.terminal = False
        """Reset the cliff, put the agent at random point
        """
        r = random.uniform(0, 1)
        if r < 0.4:
            self.current_p = [4, 1]
        else:
            r = np.random.random_integers(1, 3)
            c = np.random.random_integers(1, 6)
            self.current_p = [r, c]
        return self.current_p

    def update_current_p(self, index, incre):
        self.current_p[index] = self.current_p[index]+incre

    def move(self, action):
        """move the agent following action.

        :param action:
        :return:
            self.terminal
            self.reward
        """
        old_p = [self.current_p[0], self.current_p[1]]
        if action == 0:
            self.update_current_p(0, -1)
        if action == 1:
            self.update_current_p(0, 1)
        if action == 2:
            self.update_current_p(1, -1)
        if action == 3:
            self.update_current_p(1, 1)

        if self.current_p[0] % 5 == 0 or self.current_p[1] % 7 == 0:
            self.current_p = old_p
            self.terminal = False
            self.reward = -1
        else:
            if self.current_p == [4, 6]:
                self.terminal = True
                self.reward = 100
            elif self.current_p[0] == 4 and self.current_p[1] != 1:
                self.terminal = True
                self.reward = -100
            else:
                self.terminal = False
                self.reward = -1

        return old_p, self.terminal, self.reward, self.current_p


class Brain(object):
    """class brain (RL).
    """
    def __init__(self):
        self.Q_table_1 = np.zeros((24, 4))
        self.Q_table_2 = np.zeros((24, 4))

    def map_p_to_row(self, current_p):
        """A map from current_p to row(state index) in Q_table.
        """
        row = (current_p[0]-1)*6+(current_p[1]-1)
        return row

    def epsilon_greedy_action(self, row, episode):
        """Epsilon greedy policy for selecting action.
        """
        r = random.uniform(0, 1)
        epsilon = EPSILON
        if episode % EPSILON_DECAY_LINE == 0 and epsilon > 0.1:
            epsilon = epsilon/2
        if r < epsilon:
            action = np.random.random_integers(0, 3)
        else:
            state_values = self.Q_table_1[row, :] + self.Q_table_2[row, :]
            num = list(state_values).count(state_values.max())
            if num == 1:
                action = np.argmax(state_values)
            else:
                # When there are more than one max q, we choose one from them randomly.
                max_action_index = np.where(state_values == state_values.max())
                list_q = list(max_action_index[0])
                action = random.choice(list_q)
        return action


def double_q_learning(if_random_init):
    the_brain = Brain()
    the_cliff = Cliff()
    for i in range(MAX_EPISODES+1):
        if if_random_init:
            p = the_cliff.random_reset_cliff()
        else:
            p = the_cliff.reset_cliff()
        t = the_cliff.terminal
        s = the_cliff.current_p
        while t is False:
            row = the_brain.map_p_to_row(s)
            a = the_brain.epsilon_greedy_action(row, i)
            s, t, r, s_ = the_cliff.move(a)
            row_ = the_brain.map_p_to_row(s_)
            random_num = random.uniform(0, 1)
            if random_num < 0.5:
                select_action_s_ = np.argmax(the_brain.Q_table_1[row_, :])
                TD_error = r + GAMMA * the_brain.Q_table_2[row_, select_action_s_] - the_brain.Q_table_1[row, a]
                the_brain.Q_table_1[row, a] = the_brain.Q_table_1[row, a] + ALPHA * TD_error
            else:
                select_action_s_ = np.argmax(the_brain.Q_table_2[row_, :])
                TD_error = r + GAMMA * the_brain.Q_table_1[row_, select_action_s_] - the_brain.Q_table_2[row, a]
                the_brain.Q_table_2[row, a] = the_brain.Q_table_2[row, a] + ALPHA * TD_error
            s = s_  # I have done this in Cliff.move(), so this line can be removed.
        if i % 10000 == 0:
            print('Episode {0} end with reward:{1}'.format(i, r))
        if i % 100000 == 0:
            with open('./double_q_learning/'+str(i)+'_Q_table_dql.txt', 'w') as file:
                file.write(str((the_brain.Q_table_1+the_brain.Q_table_2)/2))
    print((the_brain.Q_table_1+the_brain.Q_table_2)/2)


def main():
    double_q_learning(True)


if __name__ == '__main__':
    main()