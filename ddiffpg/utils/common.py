import ast
import random
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from copy import deepcopy
import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_cfg(cfg, if_ddiffpg=True):

    if cfg.env.name == 'antmaze-v1':
        cfg.env.resolution = 255
        cfg.env.random_init = True
        cfg.max_step = 3000000
        if if_ddiffpg:
            cfg.algo.cluster_threshold = 50
    elif cfg.env.name == 'antmaze-v2':
        cfg.env.resolution = 357
        cfg.max_step = 3000000
        if if_ddiffpg:
            cfg.algo.cluster_threshold = 70
    elif cfg.env.name == 'antmaze-v3':
        cfg.env.resolution = 459
        cfg.max_step = 4000000
        if if_ddiffpg:
            cfg.algo.cluster_threshold = 70
    elif cfg.env.name == 'antmaze-v4':
        cfg.env.resolution = 357
        cfg.max_step = 5000000
        if if_ddiffpg:
            cfg.algo.cluster_threshold = 50
    else:
        cfg.algo.use_downsampling = True
        cfg.max_step = 3000000
    return cfg

def load_class_from_path(cls_name, path):
    mod_name = 'MOD%s' % cls_name
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[cls_name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, cls_name)


def set_random_seed(seed=None):
    if seed is None:
        max_seed_value = np.iinfo(np.uint32).max
        min_seed_value = np.iinfo(np.uint32).min
        seed = random.randint(min_seed_value, max_seed_value)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    return seed


def set_print_formatting():

    configs = dict(
        precision=6,
        edgeitems=30,
        linewidth=1000,
        threshold=5000,
    )
    np.set_printoptions(suppress=True,
                        formatter=None,
                        **configs)
    torch.set_printoptions(sci_mode=False, **configs)


def pathlib_file(file_name):
    if isinstance(file_name, str):
        file_name = Path(file_name)
    elif not isinstance(file_name, Path):
        raise TypeError(f'Please check the type of the filename:{file_name}')
    return file_name


def list_class_names(dir_path):


    dir_path = pathlib_file(dir_path)
    py_files = list(dir_path.rglob('*.py'))
    py_files = [f for f in py_files if f.is_file() and f.name != '__init__.py']
    cls_name_to_path = dict()
    for py_file in py_files:
        with py_file.open() as f:
            node = ast.parse(f.read())
        classes_in_file = [n for n in node.body if isinstance(n, ast.ClassDef)]
        cls_names_in_file = [c.name for c in classes_in_file]
        for cls_name in cls_names_in_file:
            cls_name_to_path[cls_name] = py_file
    return cls_name_to_path


class Tracker:
    def __init__(self, max_len):
        self.moving_average = deque([0 for _ in range(max_len)], maxlen=max_len)
        self.max_len = max_len

    def __repr__(self):
        return self.moving_average.__repr__()

    def update(self, value):
        if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
            self.moving_average.extend(value.tolist())
        elif isinstance(value, Sequence):
            self.moving_average.extend(value)
        else:
            self.moving_average.append(value)

    def mean(self):
        return np.mean(self.moving_average)

    def std(self):
        return np.std(self.moving_average)

    def max(self):
        return np.max(self.moving_average)


def get_action_dim(action_space):
    if isinstance(action_space, gym.spaces.Discrete):
        act_size = action_space.n
    elif isinstance(action_space, gym.spaces.Box):
        act_size = action_space.shape[0]
    else:
        raise TypeError
    return act_size


def normalize(input, normalize_tuple):
    if normalize_tuple is not None:
        current_mean, current_var, epsilon = normalize_tuple
        y = (input - current_mean.float()) / torch.sqrt(current_var.float() + epsilon)
        y = torch.clamp(y, min=-5.0, max=5.0)
        return y
    return input


def capture_keyboard_interrupt():
    import signal
    import sys
    def signal_handler(signal, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)


def handle_timeout(dones, info):
    timeout_key = 'time_outs'
    timeout_envs = None
    if timeout_key in info:
        timeout_envs = info[timeout_key]
    if timeout_envs is not None:
        dones = torch.logical_xor(dones, timeout_envs).type(torch.float32)

    return dones


def aggregate_traj_info(infos, key, single_info=False):
    if single_info:
        infos = [infos]
    if isinstance(infos[0], Sequence):
        out = []
        for info in infos:
            time_out = []
            for env_info in info:
                time_out.append(env_info[key])
            out.append(np.stack(time_out))
        out = stack_data(out)
    elif isinstance(infos[0], dict):
        out = []
        for info in infos:
            tensor = info[key]
            out.append(tensor)
        out = stack_data(out)
    else:
        raise NotImplementedError
    if single_info:
        out = out.squeeze(0)
    return out


def stack_data(data, torch_to_numpy=False, dim=0):
    if isinstance(data[0], dict):
        out = dict()
        for key in data[0].keys():
            out[key] = stack_data([x[key] for x in data], dim=dim)
        return out
    try:
        ret = torch.stack(data, dim=dim)
        if torch_to_numpy:
            ret = ret.cpu().numpy()
    except:

        ret = data
    return ret


class DensityTracker:
    def __init__(self, env_kwargs, resolution, type='coverage'):
        self.env_kwargs = env_kwargs
        self.maze_map = env_kwargs['maze_map']
        self.maze_size = env_kwargs['maze_size_scaling']
        self.res = resolution
        self.type = type
        self.mat, self.reset = self.generate_mat(env_kwargs['maze_map'])
        self.num_entries = (self.mat == 0).sum()

    def generate_mat(self, maze_map):
        scaled_mat = torch.zeros((self.res, self.res))
        assert self.res % len(maze_map) == 0
        scale = self.res // len(maze_map)
        for i in range(len(maze_map)):
            for j in range(len(maze_map[i])):
                for n in range(scale):
                    for m in range(scale):
                        if maze_map[i][j] == 1:
                            scaled_mat[i*scale+m][j*scale+n] = -1
                        elif maze_map[i][j] == 'g':
                            pass
                        elif maze_map[i][j] == 0:
                            pass
                        elif maze_map[i][j] == 'r':
                            reset = (i, j)
                        else:
                            assert 0
        reset = [reset[0]*scale+scale//2, reset[1]*scale+scale//2]
        return scaled_mat, reset

    def convert_pos_to_idx(self, pos):

        idx = pos / self.maze_size * (self.res // len(self.maze_map))
        idx_x = (-idx[:, 1] + self.reset[0]).type(torch.int64)
        idx_y = (idx[:, 0] + self.reset[1]).type(torch.int64)
        return idx_x, idx_y

    def update_mat(self, pos, value=None):
        idx_x, idx_y = self.convert_pos_to_idx(pos)
        if value is not None:
            assert value.shape[0] == pos.shape[0]
            self.mat[idx_x, idx_y] = value
        else:
            self.mat[idx_x, idx_y] += 1

    def plot_heatmap(self):
        fig, ax = plt.subplots()
        mat = deepcopy(self.mat).numpy()
        mat[mat==-1] = 0
        with sns.axes_style("white"):
            if self.type == 'qvalue':
                ax = sns.heatmap(mat, vmax=10, square=True, cmap='Reds', linewidths=0, rasterized=True)
            else:
                ax = sns.heatmap(mat, vmax=100, square=True, cmap='Reds', linewidths=0, rasterized=True)
            ax.set(xticklabels=[])
            ax.set(yticklabels=[])
            fig.canvas.draw()
            image_flat = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')

            image = image_flat.reshape(*reversed(fig.canvas.get_width_height()), 3)
            plt.close()
        return image

    def get_density(self):
        return (self.mat > 0).sum() / self.num_entries
