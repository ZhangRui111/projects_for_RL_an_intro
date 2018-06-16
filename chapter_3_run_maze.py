import numpy as np

MAX_STEPS = 100
GAMMA = 0.9


# def move(action, maze, loc):
#     """Take action when agent is in location(loc) of the maze and return the new maze and location.
#
#     :param action:
#     :param maze:
#     :param loc:
#     :return:
#     """
#     new_loc = loc
#
#     if loc == (0, 1):
#         new_loc = (4, 1)
#         reward = 10
#     elif loc == (0, 3):
#         new_loc = (2, 3)
#         reward = 5
#     else:
#         if action == 0:
#             new_loc = (loc[0] - 1, loc[1])
#         elif action == 1:
#             new_loc = (loc[0], loc[1] - 1)
#         elif action == 2:
#             new_loc = (loc[0] + 1, loc[1])
#         else:
#             new_loc = (loc[0], loc[1] + 1)
#
#         if new_loc[0] < 0 or new_loc[1] < 0 or new_loc[0] > 4 or new_loc[1] > 4:
#             new_loc = loc
#             reward = -1
#         else:
#             reward = 0
#
#     loc_value = update(loc, reward, new_loc)
#     maze[loc] = loc_value
#
#     return maze, new_loc


def update(loc, maze):
    if loc == (0, 1):
        maze[loc] = 10 + maze[(4, 1)]
        return maze
    if loc == (0, 3):
        maze[loc] = 5 + maze[(2, 3)]
        return maze

    if loc == (0, 0):
        maze[loc] = 1 / 4 * (-1 + -1 + maze[(0, 1)] + maze[(1, 0)])
    elif loc == (0, 4):
        maze[loc] = 1 / 4 * (-1 + -1 + maze[(0, 3)] + maze[(1, 4)])
    elif loc == (4, 0):
        maze[loc] = 1 / 4 * (-1 + -1 + maze[(3, 0)] + maze[(4, 1)])
    elif loc == (4, 4):
        maze[loc] = 1 / 4 * (-1 + -1 + maze[(3, 4)] + maze[(4, 3)])
    elif loc[0] == 0 and 0 < loc[1] < 4:
        maze[loc] = 1 / 4 * (-1 + maze[(loc[0], loc[1] - 1)] + maze[(loc[0], loc[1] + 1)] + maze[(loc[0] + 1, loc[1])])
    elif loc[0] == 4 and 0 < loc[1] < 4:
        maze[loc] = 1 / 4 * (-1 + maze[(loc[0], loc[1] - 1)] + maze[(loc[0], loc[1] + 1)] + maze[(loc[0] - 1, loc[1])])
    elif loc[1] == 0 and 0 < loc[0] < 4:
        maze[loc] = 1 / 4 * (-1 + maze[(loc[0] + 1, loc[1])] + maze[(loc[0] - 1, loc[1])] + maze[(loc[0], loc[1] + 1)])
    elif loc[1] == 4 and 0 < loc[0] < 4:
        maze[loc] = 1 / 4 * (-1 + maze[(loc[0] + 1, loc[1])] + maze[(loc[0] - 1, loc[1])] + maze[(loc[0], loc[1] - 1)])
    else:
        maze[loc] = 1 / 4 * (maze[(loc[0], loc[1] - 1)] + maze[(loc[0], loc[1] + 1)] +
                             maze[(loc[0] - 1, loc[1])] + maze[(loc[0] + 1, loc[1])])

    return maze


if __name__ == '__main__':
    maze = np.zeros((5, 5))
    print(maze)
    for i in range(MAX_STEPS):
        if i % 5 == 0:
            print('Episode' + str(i))
            print(maze)
        for j in range(5):
            for k in range(5):
                maze = update((j, k), maze)
    print(maze)

# Episode 100
# [[ 2.25315056  9.59258725  4.52436579  6.24455993  1.27121523]
#  [ 1.420015    3.97097867  3.26031599  2.89743421  0.84030099]
#  [ 0.45593077  1.61099644  1.6484853   1.24455993  0.19255453]
#  [-0.20728837  0.36859103  0.47806881  0.2397657  -0.31464282]
#  [-0.65367528 -0.40741275 -0.34456677 -0.44892314 -0.69089149]]
