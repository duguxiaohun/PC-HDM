import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import dendrogram


colors = ['#808080', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
            '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff',
            '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
            '#000075', '#ffffff', '#000000']

def plot_cluster(kwargs, traj, clusters):
    maze_map = kwargs['maze_map']
    maze_size = kwargs['maze_size_scaling']

    start = None
    goals = []
    blocks = []

    for i in range(len(maze_map)):
        for j in range(len(maze_map[0])):
            if maze_map[i][j] == 'r':
                start = (i, j)
            elif maze_map[i][j] == 'g':
                goals.append((i, j))
            elif maze_map[i][j] == 1:
                blocks.append((i, j))

    fig, ax = plt.subplots()


    x_lim = (-(start[1] + 0.5) * maze_size, (len(maze_map[0]) - 0.5 - start[1]) * maze_size)
    y_lim = (-(len(maze_map[0]) - 0.5 - start[0]) * maze_size, (start[0] + 0.5) * maze_size)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)


    for block in blocks:
        x, y = x_lim[0] + maze_size * block[1], maze_size * block[0] - y_lim[1]
        ax.add_patch(Rectangle((x, y), maze_size, maze_size, linewidth=0, rasterized=True, color='#C0C0C0'))


    for i in range(len(clusters)):
        points = []
        for j in range(len(clusters[i])):
            points.append(traj[clusters[i][j]])
        if len(points) != 0:
            points = np.concatenate(points)
            ax.scatter(points[:, 0], points[:, 1], s=0.01, color=colors[i])


    ax.plot(0, 0, 'ro')
    ax.annotate('start', (0, 0.25))
    for goal in goals:
        x = (goal[1] - start[1]) * maze_size
        y = (goal[0] - start[0]) * maze_size
        ax.plot(x, y, 'bo')
        ax.annotate('goal', (x, y + 0.25))


    fig.canvas.draw()
    image_flat = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')

    image = image_flat.reshape(*reversed(fig.canvas.get_width_height()), 3)
    plt.close()
    return image


def plot_hierarchy(Z):
    fig, ax = plt.subplots()
    dn = dendrogram(Z)
    fig.canvas.draw()
    image_flat = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')

    image = image_flat.reshape(*reversed(fig.canvas.get_width_height()), 3)
    plt.close()
    return image


def plot_traj(kwargs, points):
    maze_map = kwargs['maze_map']
    maze_size = kwargs['maze_size_scaling']

    start = None
    goals = []
    blocks = []

    for i in range(len(maze_map)):
        for j in range(len(maze_map[0])):
            if maze_map[i][j] == 'r':
                start = (i, j)
            elif maze_map[i][j] == 'g':
                goals.append((i, j))
            elif maze_map[i][j] == 1:
                blocks.append((i, j))

    fig, ax = plt.subplots()


    x_lim = (-(start[1] + 0.5) * maze_size, (len(maze_map[0]) - 0.5 - start[1]) * maze_size)
    y_lim = (-(len(maze_map[0]) - 0.5 - start[0]) * maze_size, (start[0] + 0.5) * maze_size)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)


    for block in blocks:
        x, y = x_lim[0] + maze_size * block[1], maze_size * block[0] - y_lim[1]
        ax.add_patch(Rectangle((x, y), maze_size, maze_size, linewidth=0, rasterized=True, color='#C0C0C0'))


    ax.scatter(points[:, 0], points[:, 1], s=0.01, color=colors[0])


    ax.plot(0, 0, 'ro')
    ax.annotate('start', (0, 0.25))
    for goal in goals:
        x = (goal[1] - start[1]) * maze_size
        y = (goal[0] - start[0]) * maze_size
        ax.plot(x, y, 'bo')
        ax.annotate('goal', (x, y + 0.25))


    fig.canvas.draw()
    image_flat = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')

    image = image_flat.reshape(*reversed(fig.canvas.get_width_height()), 3)
    plt.close()
    return image
