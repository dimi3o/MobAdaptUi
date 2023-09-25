import numpy as np

from dqnvianumpy.helper import format_matrix

class FrozenLake:
    def __init__(self):
        #1: starting point, 2: frozen surface, 3: hole, 4: goal
        self.gameboard = np.array([1, 2, 2, 2,
                                   2, 3, 2, 3,
                                   2, 2, 2, 3,
                                   3, 2, 2, 4])
        self.state = np.array([1, 0, 0, 0,
                               0, 0, 0, 0,
                               0, 0, 0, 0,
                               0, 0, 0, 0])
        self.position = 0

    def reset(self):
        """
        method resets environment
        """
        self.position = 0
        self.state = np.array([1, 0, 0, 0,
                               0, 0, 0, 0,
                               0, 0, 0, 0,
                               0, 0, 0, 0])
        return self.state

    def render(self, console):
        """
        method renders environment
        """
        string = ""
        fields = ["S", "F", "F", "F", "F", "H", "F", "H", "F", "F", "F", "H", "H", "F", "F", "G"]
        for i, item in enumerate(self.state):
            if item == 1:
                string = string + '(' + fields[i] + ')'
            else:
                string = string + fields[i]
            if i in [3, 7, 11, 15]:
                string = string + '\n'
        console.cwriteline(string)

    def step(self, action):
        """
        method makes step in environment
        """
        #0: up, 1: left, 2:down, 3: right
        reward, done = self.get_rw_dn()
        if done:
            return self.state, reward, done

        if action == 0:
            if self.position in [0,1,2,3]:
                return self.state, reward, done
            else:
                self.position -= 4

        elif action == 1:
            if self.position in [3,7,11,15]:
                return self.state, reward, done
            else:
                self.position += 1

        elif action == 2:
            if self.position in [12,13,14,15]:
                return self.state, reward, done
            else:
                self.position += 4

        elif action == 3:
            if self.position in [0,4,8,12]:
                return self.state, reward, done
            else:
                self.position -= 1

        self.state = np.zeros(16)
        self.state[self.position] = 1

        reward, done = self.get_rw_dn()

        return self.state, reward, done

    def get_rw_dn(self):
        """
        method returns reward and done status for actual state
        """
        reward = -1
        done = False
        if self.gameboard[self.position] in [1,2]:
            reward = 0
        elif self.gameboard[self.position] == 3:
            reward = -1
            done = True
        elif self.gameboard[self.position] == 4:
            reward = 1
            done = True

        return reward, done

    def render_wQ(self, q_values, console):
        """
        method renders environment with actual q-values
        """
        console.cwriteline(''+str(q_values[0][3])+','+str(q_values[1][3])+','+str(q_values[2][3])+','+str(q_values[3][3])+','+str(q_values[4][3])+','+str(q_values[5][3])+','+str(q_values[6][3])+','+str(q_values[7][3])+','+str(q_values[8][3])+','+str(q_values[9][3])+','+str(q_values[10][3])+','+str(q_values[11][3])+','+str(q_values[12][3])+','+str(q_values[13][3])+','+str(q_values[14][3])+','+str(q_values[15][3])+'\n')
        return
        # string = format_matrix(['','0','', '','1','', '','2','','', '3',''],
        #                     [[42, q_values[0][0], 42, 42, q_values[1][0], 42, 42, q_values[2][0], 42, 42, q_values[3][0], 42],
        #                      [q_values[0][3], 42.001, q_values[0][1], q_values[1][3], 42.002, q_values[1][1], q_values[2][3], 42.002, q_values[2][1], q_values[3][3], 42.002, q_values[3][1]],
        #                      [42, q_values[0][2], 42, 42, q_values[1][2], 42, 42, q_values[2][2], 42, 42, q_values[3][2], 42],
        #                      [42, q_values[4][0], 42, 42, q_values[5][0], 42, 42, q_values[6][0], 42, 42, q_values[7][0], 42],
        #                      [q_values[4][3], 42.002, q_values[4][1], q_values[5][3], 42.003, q_values[5][1], q_values[6][3], 42.002, q_values[6][1], q_values[7][3], 42.003, q_values[7][1]],
        #                      [42, q_values[4][2], 42, 42, q_values[5][2], 42, 42, q_values[6][2], 42, 42, q_values[7][2], 42],
        #                      [42, q_values[8][0], 42, 42, q_values[9][0], 42, 42, q_values[10][0], 42, 42, q_values[11][0], 42],
        #                      [q_values[8][3], 42.002, q_values[8][1], q_values[9][3], 42.002, q_values[9][1], q_values[10][3], 42.002, q_values[10][1], q_values[11][3], 42.003, q_values[11][1]],
        #                      [42, q_values[8][2], 42, 42, q_values[9][2], 42, 42, q_values[10][2], 42, 42, q_values[11][2], 42],
        #                      [42, q_values[12][0], 42, 42, q_values[13][0], 42, 42, q_values[14][0], 42, 42, q_values[15][0], 42],
        #                      [q_values[12][3], 42.003, q_values[12][1], q_values[13][3], 42.002, q_values[13][1], q_values[14][3], 42.002, q_values[14][1], q_values[15][3], 42.004, q_values[15][1]],
        #                      [42, q_values[12][2], 42, 42, q_values[13][2], 42, 42, q_values[14][2], 42, 42, q_values[15][2], 42]],
        #                     '{:^{}}', '{:<{}}', '{:>{}.3f}', '\n', ' | ')
        # string = string.replace('42.001', '   S  ')
        # string = string.replace('42.002', '   F  ')
        # string = string.replace('42.003', '   H  ')
        # string = string.replace('42.004', '   G  ')
        # console.cwriteline(string.replace('42.000', '      '))
