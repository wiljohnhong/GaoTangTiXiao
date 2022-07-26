import collections
import copy
import functools
import hashlib
import math
import os
import pickle
from collections import Sequence

import torch
import torch.nn as nn
import numpy as np

N_PLAYER = 2

POSSIBLE_INIT_MD5 = {
    # running 1 (straight), team 0
    b'\xeca\xb6A\x808\xe1\x08)F\xf4gmW\xad1': 'running-straight',
    # running 1 (straight), team 1
    b'&[\xb9\xcb\xe3\xba7\xc9\xdc\xbfT\xdb\xf6_\xd3\x85': 'running-straight',
    # running 2 (whistle), team 0
    b'\x06&\x95\x1a\xab\xd3n\xd2\xcfh{\x10#\xaa\xfc\x10': 'running-whistle',
    # running 2 (whistle), team 1
    b'\xe4@\xb66w\xcb\xbbZV\xf5K0GzZ\xfd': 'running-whistle',
    # running 3 (T_shape), team 0
    b'2d\x85\xa3\x7fTO\xcd\xa7\x15\xac!\x04\xdc\x02\xb8': 'running-T_shape',
    b'\xa0\x8b\x0b1\xe1\xad\x87\xd1?\x83\xa5,_\xbe\xb7\x1b': 'running-T_shape',
    # running 3 (T_shape), team 1
    b'\x1a\x98\xbb\xae\x04a\xe2(!\xe8\xea>\x11\xac-\x87': 'running-T_shape',
    # running 4 (square), team 0
    b'\\w\x81\xbch\xf1\xc3\x16\xd3\xca\x04h\xb2\xbd\xe13': 'running-square',
    # running 4 (square), team 1
    b'G(\xd1v\xc2\x89\n\x00\xddnhG\xaf\xeem\xf2': 'running-square',
    # table-hockey, team 0
    b'\xdb?\x7f\x15\xe4\x8a\xf6~*\xca\xa6\x05\x95<\xef\xbb': 'table-hockey',
    # table-hockey team 1
    b'N\xa1\xb6\xe3;\x93"\x9f\xc4\x07<\x9b\xf4\x96`A': 'table-hockey',
    # football, team 0
    b'\x90}\xbd\x0e\x03\xe8\x97\xf3\xbf\xde\xaf\xb3b\xb0#\x13': 'football',
    # football, team 1
    b'\x9bl\n\xe9w\xc6m\x00\x08\xb8\x83\x14\xe2s2\xcf': 'football',
    # wrestling, team 0
    b"\xfa\xb5\xa4v\x1a\x01\x9eNij\xfe$d'\xf2\x83": 'wrestling',
    # wrestling, team 1
    b'E\xc3r\xa8\x04q\xc9S\xe5\x1e\x0eaV\xe7\x91x': 'wrestling',
    # curling, team 0
    b'\x0eiIq\xe7\x11^\xf5\xf5 \xa2\xea\xe6\xd4e\xf7': 'curling',
    # curling, team 1
    b'\xfd\x8c\xce\xce\xe7P\x07\xc6\x92@\xbe;K\x8f\xa5\xca': 'curling',
    # billiard, team 0
    b'\xb5m\xbel\xa4\x14\x91\x15\xe7\xc2\xcb\xfcz4d\x95': 'billiard',
    # billiard, team 1
    b'\xe3\x01\x02wqv4\x80\xd8)\xd1\xcba\xc4=\xce': 'billiard',
}


def one_hot_generator(n_feature, index):
    one_hot = np.zeros(n_feature,)
    one_hot[index] = 1
    return one_hot


def multi_hot_generator(n_feature, index):
    one_hot = np.zeros(n_feature,)
    one_hot[:index] = 1
    return one_hot


def cannot_act(ob):
    """Cannot act in curling."""
    return (ob['agent_obs'] == -1.).all()


class OldGeneralTranslator:
    LEN_IMG_HIST = 64
    LEN_ACT_HIST = 64

    def __init__(self):
        self.img_history = None
        self.act_history = None
        self.team_id = None
        self.subgame_step = None
        self.agent_theta = None

        # curling specialized
        self.step_since_inner_start = None
        self.prev_obs = None
        self.prev_cannot_act = None
        self.round_cnt = None

        self.init_obs = None
        self.init_angle = None

    def reset(self, init_obs):
        self.team_id = int(init_obs['id'][-1])
        self.subgame_step = 0
        self.agent_theta = 0.

        # curling specialized
        self.step_since_inner_start = -1
        self.prev_obs = init_obs
        self.prev_cannot_act = True
        self.round_cnt = 0

        self.img_history = collections.deque(maxlen=self.LEN_IMG_HIST)
        for _ in range(self.LEN_IMG_HIST):
            self.img_history.append(np.zeros((40, 40)))

        self.act_history = collections.deque(maxlen=self.LEN_ACT_HIST)
        for _ in range(self.LEN_ACT_HIST):
            self.act_history.append([0., 0.])

        self.init_obs = init_obs
        self.init_angle = np.random.choice([0.55, 0.6, 0.65, 0.7, 0.75, 0.8])

    def trans_obs(self, obs):
        self.subgame_step += 1

        if cannot_act(obs):
            self.prev_cannot_act = True
            self.prev_obs = obs
            return
        else:
            # take care of the annoying case when agent1's round2 starts
            if self.prev_cannot_act or obs.get('info', None) == 'Reset Round':
                self.round_cnt += 1
                if self.round_cnt > 3:
                    self.reset(obs)
                    self.round_cnt += 1
                self.step_since_inner_start = -1
                self.agent_theta = 0.
            self.prev_obs = obs
            self.prev_cannot_act = False
            self.step_since_inner_start += 1

        self.img_history.append(obs['agent_obs'])

        # [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 57, 58, 59, 60, 61, 62, 63]
        # 0th is the oldest frame, 63th is the latest frame
        img = np.stack(self.img_history) / 10.
        img = np.concatenate([img[:56:4], img[56:]])

        angle = self.agent_theta / 180. * np.pi
        game_progress = self.subgame_step / 600.
        inner_progress = (self.step_since_inner_start % 110) / 110.

        vec = np.concatenate([
            one_hot_generator(n_feature=N_PLAYER, index=self.team_id),
            one_hot_generator(n_feature=3, index=self.round_cnt-1),
            multi_hot_generator(n_feature=12, index=int(game_progress * 12) + 1),
            multi_hot_generator(n_feature=10, index=int(inner_progress * 10) + 1),
            [game_progress, inner_progress],
            [obs['energy'] < 100., obs['energy'] / 1000.],
            [np.cos(angle), np.sin(angle)],
            np.concatenate(self.act_history),
        ])

        if 'cheat' in obs:
            cheat = obs['cheat']
        else:
            cheat = np.zeros((64,))

        legal = np.ones((110,))
        if obs.get('line_past', False):
            legal = np.zeros((110,))
            legal[38] = 1  # zero force & zero angle

        obs = {'img': img, 'vec': vec, 'cheat': cheat, 'legal': legal}
        return obs

    def trans_action(self, action):
        force = [-100, -67, -33, 0, 33, 67, 100, 133, 167, 200][action // 11]
        angle = [-30, -24, -18, -12, -6, 0, 6, 12, 18, 24, 30][action % 11]
        atom_action = [force, angle]
        self.agent_theta = (self.agent_theta + angle) % 360
        self.act_history.append([force / 200., angle / 30.])
        return atom_action


class NewGeneralTranslator:
    LEN_IMG_HIST = 64
    LEN_ACT_HIST = 64

    def __init__(self):
        self.img_history = None
        self.act_history = None
        self.team_id = None
        self.subgame_step = None
        self.agent_theta = None
        self.scenario = None

        # curling legacy
        self.step_since_inner_start = None
        self.round_cnt = None

    def reset(self, init_obs):
        self.team_id = int(init_obs['id'][-1])
        self.subgame_step = 0
        self.agent_theta = 0.

        init_md5 = hashlib.md5(init_obs['agent_obs'].tobytes()).digest()
        self.scenario = POSSIBLE_INIT_MD5[init_md5]

        self.step_since_inner_start = -1
        self.round_cnt = 1

        self.img_history = collections.deque(maxlen=self.LEN_IMG_HIST)
        for _ in range(self.LEN_IMG_HIST):
            self.img_history.append(np.zeros((40, 40)))

        self.act_history = collections.deque(maxlen=self.LEN_ACT_HIST)
        for _ in range(self.LEN_ACT_HIST):
            self.act_history.append([0., 0.])

    def trans_obs(self, obs):
        self.subgame_step += 1
        self.step_since_inner_start += 1

        self.img_history.append(obs['agent_obs'])

        # [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 57, 58, 59, 60, 61, 62, 63]
        # 0th is the oldest frame, 63th is the latest frame
        img = np.stack(self.img_history) / 10.
        img = np.concatenate([img[:56:4], img[56:]])

        angle = self.agent_theta / 180. * np.pi
        game_progress = self.subgame_step / 600.
        inner_progress = (self.step_since_inner_start % 110) / 110.

        vec = np.concatenate([
            one_hot_generator(n_feature=N_PLAYER, index=self.team_id),
            one_hot_generator(n_feature=3, index=self.round_cnt-1),
            multi_hot_generator(n_feature=12, index=int(game_progress * 12) + 1),
            multi_hot_generator(n_feature=10, index=int(inner_progress * 10) + 1),
            [game_progress, inner_progress],
            [obs['energy'] < 100., obs['energy'] / 1000.],
            [np.cos(angle), np.sin(angle)],
            np.concatenate(self.act_history),
        ])

        if 'cheat' in obs:
            cheat = obs['cheat']
        else:
            cheat = np.zeros((64,))

        legal_force = np.ones((61,))
        legal_angle = np.ones((61,))
        if obs.get('line_past', False):
            legal_force = np.zeros((61,))
            legal_force[20] = 1
            legal_angle = np.zeros((61,))
            legal_angle[30] = 1

        obs = {'img': img, 'vec': vec, 'cheat': cheat, 'legal_force': legal_force, 'legal_angle': legal_angle}
        return obs

    def trans_action(self, action):
        act_force, act_angle = action[0], action[1]
        force = act_force * 5 - 100
        angle = act_angle - 30

        atom_action = [force, angle]
        self.agent_theta = (self.agent_theta + angle) % 360
        self.act_history.append([force / 200., angle / 30.])
        return atom_action


left2_pre = [[-88, -6.5]] * 20 + [[176, 13]] * 10
left1_pre = [[-77, -4.3]] * 18 + [[154, 8.6]] * 9
middle_pre = [[-30, 0]] + [[-100, 0]] * 14 + [[200, 0]] * 7
right1_pre = [[-77, 4.3]] * 18 + [[154, -8.6]] * 9
right2_pre = [[-88, 6.5]] * 20 + [[176, -13]] * 10


class CurlingTranslator:
    LEN_IMG_HIST = 16
    LEN_ACT_HIST = 16

    def __init__(self):
        self.img_history = None
        self.act_history = None
        self.team_id = None
        self.subgame_step = None
        self.agent_theta = None

        # curling specialized
        self.step_since_inner_start = None
        self.prev_obs = None
        self.prev_cannot_act = None
        self.round_cnt = None
        self.curling_center_img = None
        self.macro_chosen = None
        self.micro_actions_left = None
        self.acc = None
        self.acc_actions = None

    def reset(self, init_obs):
        self.team_id = int(init_obs['id'][-1])
        self.subgame_step = 0
        self.agent_theta = 0.

        # curling specialized
        self.step_since_inner_start = -1
        self.prev_obs = init_obs
        self.prev_cannot_act = True
        self.round_cnt = 0
        self.curling_center_img = None
        self.macro_chosen = None
        self.micro_actions_left = None
        self.acc = False
        self.acc_actions = collections.deque([[-100, 0]] * 14 + [[200, 0.05]] * 200)

        self.img_history = collections.deque(maxlen=self.LEN_IMG_HIST)
        for _ in range(self.LEN_IMG_HIST):
            self.img_history.append(np.zeros((40, 40)))

        self.act_history = collections.deque(maxlen=self.LEN_ACT_HIST)
        for _ in range(self.LEN_ACT_HIST):
            self.act_history.append([0., 0.])

    def trans_obs(self, obs):
        self.subgame_step += 1

        if cannot_act(obs):
            self.prev_cannot_act = True
            self.prev_obs = obs
            return
        else:
            # take care of the annoying case when agent1's round2 starts
            if self.prev_cannot_act or obs.get('info', None) == 'Reset Round':
                self.round_cnt += 1
                if self.round_cnt > 3:
                    self.reset(obs)
                    self.round_cnt += 1
                self.step_since_inner_start = -1
                self.agent_theta = 0.
            self.prev_obs = obs
            self.prev_cannot_act = False
            self.step_since_inner_start += 1

        if self.step_since_inner_start == 22:
            self.curling_center_img = obs['agent_obs']
            if self.round_cnt == 3:
                if self.team_id == 0 and np.sum(obs['agent_obs'][3:30, 17:22]) == 256 and np.sum(obs['agent_obs'][3:12, 17:22]) == 120:
                    self.acc = True
                if self.team_id == 1 and np.sum(obs['agent_obs'][3:30, 17:22]) == 280 and np.sum(obs['agent_obs'][3:12, 17:22]) == 144:
                    self.acc = True
        if self.step_since_inner_start < 24:
            return

        if self.step_since_inner_start == 24:
            self.img_history = collections.deque(maxlen=self.LEN_IMG_HIST)
            for _ in range(self.LEN_IMG_HIST):
                self.img_history.append(np.zeros((40, 40)))
            self.act_history = collections.deque(maxlen=self.LEN_ACT_HIST)
            for _ in range(self.LEN_ACT_HIST):
                self.act_history.append([0., 0.])

            self.micro_actions_left = [
                collections.deque(left2_pre),
                collections.deque(left1_pre),
                collections.deque(middle_pre),
                collections.deque(right1_pre),
                collections.deque(right2_pre),
            ]

        self.img_history.append(obs['agent_obs'])

        img = np.stack(self.img_history) / 10.

        angle = self.agent_theta / 180. * np.pi
        game_progress = self.subgame_step / 600.
        inner_progress = (self.step_since_inner_start % 110) / 110.

        vec = np.concatenate([
            one_hot_generator(n_feature=N_PLAYER, index=self.team_id),
            one_hot_generator(n_feature=3, index=self.round_cnt-1),
            multi_hot_generator(n_feature=12, index=int(game_progress * 12) + 1),
            multi_hot_generator(n_feature=10, index=int(inner_progress * 10) + 1),
            [game_progress, inner_progress],
            [obs['energy'] < 100., obs['energy'] / 1000.],
            [np.cos(angle), np.sin(angle)],
            np.concatenate(self.act_history),
        ])

        if 'cheat' in obs:
            cheat = obs['cheat']
        else:
            cheat = np.zeros((64,))

        legal_macro = np.zeros((6,))  # the last action denotes no-op
        legal_force = np.ones((61,))
        legal_angle = np.ones((61,))
        if self.step_since_inner_start == 24:
            legal_macro[:-1] = 1
            legal_force = np.zeros((61,))
            legal_force[20] = 1  # zero force
            legal_angle = np.zeros((61,))
            legal_angle[30] = 1  # zero angle
        else:
            legal_macro[-1] = 1
            if obs.get('line_past', False):
                legal_force = np.zeros((61,))
                legal_force[20] = 1
                legal_angle = np.zeros((61,))
                legal_angle[30] = 1

        obs = {
            'img': img,
            'center': self.curling_center_img[np.newaxis, :],
            'vec': vec,
            'cheat': cheat,
            'legal_force': legal_force,
            'legal_angle': legal_angle,
            'legal_macro': legal_macro,
        }
        return obs

    def trans_action(self, action):
        act_force, act_angle, act_macro = action[0], action[1], action[2]
        if act_macro != 5:
            self.macro_chosen = act_macro

        if not self.acc:
            if len(self.micro_actions_left[self.macro_chosen]) > 0:
                force, angle = self.micro_actions_left[self.macro_chosen].popleft()
            else:
                force = act_force * 5 - 100
                angle = act_angle - 30
        else:
            return self.acc_actions.popleft()

        atom_action = [force, angle]
        self.agent_theta = (self.agent_theta + angle) % 360
        self.act_history.append([force / 200., angle / 30.])
        return atom_action


def tensorize_state(func):
    def _recursive_processing(state, device):
        if not isinstance(state, torch.Tensor):
            if isinstance(state, dict):
                for k, v in state.items():
                    state[k] = _recursive_processing(state[k], device)
            else:
                state = torch.FloatTensor(state).to(device)
        return state

    @functools.wraps(func)
    def wrap(self, state, *arg, **kwargs):
        state = copy.deepcopy(state)
        state = _recursive_processing(state, self.device)
        return func(self, state, *arg, **kwargs)

    return wrap


class Agent:
    def __init__(self, use_gpu: bool, *args, **kwargs):
        self.use_gpu = use_gpu

        self.device = torch.device('cuda') if use_gpu else torch.device('cpu')
        self.state_handler_dict = {}

        torch.set_num_threads(1)
        self.training_iter = 0

    def register_model(self, name, model):
        assert isinstance(model, nn.Module)
        if name in self.state_handler_dict:
            raise KeyError(f"model named with {name} reassigned.")
        self.state_handler_dict[name] = model

    def loads(self, agent_dict):
        self.training_iter = agent_dict['training_iter']

        for name, np_dict in agent_dict['model_dict'].items():
            model = self.state_handler_dict[name]  # alias
            state_dict = {
                k: torch.as_tensor(v.copy(), device=self.device)
                for k, v in zip(model.state_dict().keys(), np_dict.values())
            }
            model.load_state_dict(state_dict)


def legal_mask(logit: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    mask = torch.ones_like(legal) * -math.inf
    logit = torch.where(legal == 1., logit, mask)
    return logit


class OldGeneralAgent(Agent):
    def __init__(self, use_gpu, *, net_cls, net_conf):
        super().__init__(use_gpu)
        self.net = net_cls(**net_conf).to(self.device)
        self.register_model('net', self.net)

    @tensorize_state
    def infer(self, state):
        with torch.no_grad():
            logit, value = self.net.infer(state)
            if isinstance(state, dict) and 'legal' in state:
                logit = legal_mask(logit, state['legal'])

            dist = torch.distributions.Categorical(logits=logit)
            action = dist.sample()
            action = action.item()

        return action


class NewGeneralAgent(Agent):
    def __init__(self, use_gpu, *, net_cls, net_conf,):
        super().__init__(use_gpu)
        self.net = net_cls(**net_conf).to(self.device)
        self.register_model('net', self.net)

    @tensorize_state
    def infer(self, state):
        with torch.no_grad():
            logit_force, logit_angle, value = self.net.infer(state)
            logit_force = legal_mask(logit_force, state['legal_force'])
            logit_angle = legal_mask(logit_angle, state['legal_angle'])

            dist_force = torch.distributions.Categorical(logits=logit_force)
            action_force = dist_force.sample()

            dist_angle = torch.distributions.Categorical(logits=logit_angle)
            action_angle = dist_angle.sample()

            action = (action_force.item(), action_angle.item())

        return action


class CurlingAgent(Agent):
    def __init__(self, use_gpu, *, net_cls, net_conf):
        super().__init__(use_gpu)
        self.net = net_cls(**net_conf).to(self.device)
        self.register_model('net', self.net)

    @tensorize_state
    def infer(self, state):
        with torch.no_grad():
            logit_force, logit_angle, logit_macro, value = self.net.infer(state)
            logit_force = legal_mask(logit_force, state['legal_force'])
            logit_angle = legal_mask(logit_angle, state['legal_angle'])
            logit_macro = legal_mask(logit_macro, state['legal_macro'])

            dist_force = torch.distributions.Categorical(logits=logit_force)
            action_force = dist_force.sample()

            dist_angle = torch.distributions.Categorical(logits=logit_angle)
            action_angle = dist_angle.sample()

            dist_macro = torch.distributions.Categorical(logits=logit_macro)
            action_macro = dist_macro.sample()

            action = (action_force.item(), action_angle.item(), action_macro.item())

        return action


def single_as_batch(func):
    def _recursive_processing(x, squeeze=False):
        if isinstance(x, Sequence):
            return (_recursive_processing(_, squeeze) for _ in x)
        elif isinstance(x, dict):
            return {k: _recursive_processing(v, squeeze) for k, v in x.items()}
        else:
            return x.squeeze(0) if squeeze else x.unsqueeze(0)

    @functools.wraps(func)
    def wrap(self, *tensors):
        tensors = _recursive_processing(tensors)
        result = func(self, *tensors)
        return _recursive_processing(result, squeeze=True)

    return wrap


def same_padding(in_size, filter_size, stride_size):
    in_height, in_width = in_size
    if isinstance(filter_size, int):
        filter_height, filter_width = filter_size, filter_size
    else:
        filter_height, filter_width = filter_size
    stride_height, stride_width = stride_size

    out_height = np.ceil(float(in_height) / float(stride_height))
    out_width = np.ceil(float(in_width) / float(stride_width))

    pad_along_height = int(
        ((out_height - 1) * stride_height + filter_height - in_height))
    pad_along_width = int(
        ((out_width - 1) * stride_width + filter_width - in_width))
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    padding = (pad_left, pad_right, pad_top, pad_bottom)
    output = (out_height, out_width)
    return padding, output


class SlimConv2d(nn.Module):
    """Simple mock of tf.slim Conv2d"""
    def __init__(self, in_channels, out_channels, kernel, stride, padding,
                 initializer="default", activation_fn=None, bias_init=0):
        super(SlimConv2d, self).__init__()
        layers = []

        # Padding layer.
        if padding:
            layers.append(nn.ZeroPad2d(padding))

        # Actual Conv2D layer (including correct initialization logic).
        conv = nn.Conv2d(in_channels, out_channels, kernel, stride)
        if initializer:
            if initializer == "default":
                initializer = nn.init.xavier_uniform_
            initializer(conv.weight)
        nn.init.constant_(conv.bias, bias_init)
        layers.append(conv)
        if activation_fn is not None:
            layers.append(activation_fn())

        # Put everything in sequence.
        self._model = nn.Sequential(*layers)

    def forward(self, x):
        return self._model(x)


class ResidualBlock(nn.Module):
    def __init__(self, i_channel, o_channel, in_size, kernel_size=3, stride=1):
        """two-layer residual block."""
        super().__init__()
        self._relu = nn.ReLU(inplace=True)

        padding, out_size = same_padding(in_size, kernel_size, [stride, stride])
        self._conv1 = SlimConv2d(i_channel, o_channel,
                                 kernel=3, stride=stride,
                                 padding=padding, activation_fn=None)

        padding, out_size = same_padding(out_size, kernel_size, [stride, stride])
        self._conv2 = SlimConv2d(o_channel, o_channel,
                                 kernel=3, stride=stride,
                                 padding=padding, activation_fn=None)

        self.padding, self.out_size = padding, out_size

    def forward(self, x):
        out = self._relu(x)
        out = self._conv1(out)
        out = self._relu(out)
        out = self._conv2(out)
        out += x
        return out


class ResNet(nn.Module):
    """CNN torso used in IMPALA."""

    def __init__(self, in_ch, in_size, channel_and_blocks=None):
        super().__init__()

        out_size = in_size
        conv_layers = []
        if channel_and_blocks is None:
            channel_and_blocks = [(16, 2), (32, 2), (32, 2)]

        for (out_ch, num_blocks) in channel_and_blocks:
            # Downscale
            padding, out_size = same_padding(out_size, filter_size=3,
                                             stride_size=[1, 1])
            conv_layers.append(
                SlimConv2d(in_ch, out_ch, kernel=3, stride=1, padding=padding,
                           activation_fn=None))

            padding, out_size = same_padding(out_size, filter_size=3,
                                             stride_size=[2, 2])
            conv_layers.append(nn.ZeroPad2d(padding))
            conv_layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

            # Residual blocks
            for _ in range(num_blocks):
                res = ResidualBlock(i_channel=out_ch, o_channel=out_ch,
                                    in_size=out_size)
                conv_layers.append(res)

            padding, out_size = res.padding, res.out_size
            in_ch = out_ch

        conv_layers.append(nn.ReLU(inplace=True))
        self.resnet = nn.Sequential(*conv_layers)

    def forward(self, x):
        out = self.resnet(x)
        return out


class OldGeneralNet(nn.Module):
    def __init__(self):
        super().__init__()

        in_ch = 22
        in_size = [40, 40]
        in_vec_normal = 161
        in_vec_cheat = 64
        channel_and_blocks = [[24, 2], [24, 2], [24, 2]]

        self.resnet = ResNet(in_ch, in_size, channel_and_blocks)

        sample_input = torch.zeros(1, in_ch, *in_size)
        with torch.no_grad():
            self.n_hidden = len(self.resnet(sample_input).flatten())

        self.img_fc = nn.Sequential(
            nn.Linear(self.n_hidden, 384),
            nn.ReLU(inplace=True),
        )

        self.normal_in_fc = nn.Sequential(
            nn.Linear(in_vec_normal, 256),
            nn.ReLU(inplace=True),
        )

        self.cheat_in_fc = nn.Sequential(
            nn.Linear(in_vec_cheat, 128),
            nn.ReLU(inplace=True),
        )

        self.policy = nn.Sequential(
            nn.Linear(384 + 256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 110),
        )

        self.value = nn.Sequential(
            nn.Linear(384 + 256 + 128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    @single_as_batch
    def infer(self, x):
        return self.forward(x)

    def forward(self, x):
        x_2d = x['img']
        x_normal = x['vec']
        x_cheat = x['cheat']
        hidden_resnet = self.resnet(x_2d)
        hidden_resnet = hidden_resnet.view((hidden_resnet.shape[0], -1))
        hidden_resnet = self.img_fc(hidden_resnet)
        hidden_normal = self.normal_in_fc(x_normal)
        hidden_cheat = self.cheat_in_fc(x_cheat)
        policy_hidden = torch.cat((hidden_resnet, hidden_normal), dim=1)
        value_hidden = torch.cat((hidden_resnet, hidden_normal, hidden_cheat), dim=1)
        logits = self.policy(policy_hidden)
        value = self.value(value_hidden)
        return logits, value


class NewGeneralNet(nn.Module):
    def __init__(self):
        super().__init__()

        in_ch = 22
        in_size = [40, 40]
        in_vec_normal = 161
        in_vec_cheat = 64
        channel_and_blocks = [[24, 2], [24, 2], [24, 2]]

        self.resnet = ResNet(in_ch, in_size, channel_and_blocks)

        sample_input = torch.zeros(1, in_ch, *in_size)
        with torch.no_grad():
            self.n_hidden = len(self.resnet(sample_input).flatten())

        self.img_fc = nn.Sequential(
            nn.Linear(self.n_hidden, 384),
            nn.ReLU(inplace=True),
        )

        self.normal_in_fc = nn.Sequential(
            nn.Linear(in_vec_normal, 256),
            nn.ReLU(inplace=True),
        )

        self.cheat_in_fc = nn.Sequential(
            nn.Linear(in_vec_cheat, 128),
            nn.ReLU(inplace=True),
        )

        self.policy_share = nn.Sequential(
            nn.Linear(384 + 256, 256),
            nn.ReLU(inplace=True),
        )
        self.policy_force = nn.Linear(256, 61)
        self.policy_angle = nn.Linear(256, 61)

        self.value = nn.Sequential(
            nn.Linear(384 + 256 + 128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    @single_as_batch
    def infer(self, x):
        return self.forward(x)

    def forward(self, x):
        x_2d = x['img']
        x_normal = x['vec']
        x_cheat = x['cheat']
        hidden_resnet = self.resnet(x_2d)
        hidden_resnet = hidden_resnet.view((hidden_resnet.shape[0], -1))
        hidden_resnet = self.img_fc(hidden_resnet)
        hidden_normal = self.normal_in_fc(x_normal)
        hidden_cheat = self.cheat_in_fc(x_cheat)
        policy_hidden = torch.cat((hidden_resnet, hidden_normal), dim=1)
        value_hidden = torch.cat((hidden_resnet, hidden_normal, hidden_cheat), dim=1)
        policy_hidden = self.policy_share(policy_hidden)
        logits_force = self.policy_force(policy_hidden)
        logits_angle = self.policy_angle(policy_hidden)
        value = self.value(value_hidden)
        return logits_force, logits_angle, value


class CurlingNet(nn.Module):
    def __init__(self):
        super().__init__()

        in_size = [40, 40]
        in_vec_normal = 65
        in_vec_cheat = 64

        self.resnet1 = ResNet(16, in_size, [[16, 2], [32, 2], [32, 2]])
        self.resnet2 = ResNet(1, in_size, [[16, 2], [32, 2], [32, 2]])

        sample_input1 = torch.zeros(1, 16, *in_size)
        sample_input2 = torch.zeros(1, 1, *in_size)
        with torch.no_grad():
            self.n_hidden1 = len(self.resnet1(sample_input1).flatten())
            self.n_hidden2 = len(self.resnet2(sample_input2).flatten())

        self.img_fc1 = nn.Sequential(
            nn.Linear(self.n_hidden1, 384),
            nn.ReLU(inplace=True),
        )
        self.img_fc2 = nn.Sequential(
            nn.Linear(self.n_hidden2, 384),
            nn.ReLU(inplace=True),
        )

        self.normal_in_fc = nn.Sequential(
            nn.Linear(in_vec_normal, 128),
            nn.ReLU(inplace=True),
        )

        self.cheat_in_fc = nn.Sequential(
            nn.Linear(in_vec_cheat, 128),
            nn.ReLU(inplace=True),
        )

        self.policy_share = nn.Sequential(
            nn.Linear(384 + 384 + 128, 256),
            nn.ReLU(inplace=True),
        )
        self.policy_force = nn.Linear(256, 61)
        self.policy_angle = nn.Linear(256, 61)
        self.policy_macro = nn.Linear(256, 6)

        self.value = nn.Sequential(
            nn.Linear(384 + 384 + 128 + 128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    @single_as_batch
    def infer(self, x):
        return self.forward(x)

    def forward(self, x):
        x_2d_1 = x['img']
        x_2d_2 = x['center']
        x_normal = x['vec']
        x_cheat = x['cheat']
        hidden_resnet_1 = self.resnet1(x_2d_1)
        hidden_resnet_1 = hidden_resnet_1.view((hidden_resnet_1.shape[0], -1))
        hidden_resnet_1 = self.img_fc1(hidden_resnet_1)
        hidden_resnet_2 = self.resnet2(x_2d_2)
        hidden_resnet_2 = hidden_resnet_2.view((hidden_resnet_2.shape[0], -1))
        hidden_resnet_2 = self.img_fc2(hidden_resnet_2)
        hidden_normal = self.normal_in_fc(x_normal)
        hidden_cheat = self.cheat_in_fc(x_cheat)
        policy_hidden = torch.cat((hidden_resnet_1, hidden_resnet_2, hidden_normal), dim=1)
        value_hidden = torch.cat((hidden_resnet_1, hidden_resnet_2, hidden_normal, hidden_cheat), dim=1)
        policy_hidden = self.policy_share(policy_hidden)
        logits_force = self.policy_force(policy_hidden)
        logits_angle = self.policy_angle(policy_hidden)
        logits_macro = self.policy_macro(policy_hidden)
        value = self.value(value_hidden)
        return logits_force, logits_angle, logits_macro, value


pre_act_list = [[200, 0]] * 8 + [[-90, 0]] + [[-100, 0]] * 15


class MyAI:
    def __init__(self):
        pwd = os.path.split(os.path.abspath(__file__))[0]

        self.agents = {}
        for scenario in set(POSSIBLE_INIT_MD5.values()):
            model_path = os.path.join(pwd, 'model', f'{scenario}.pkl')
            with open(model_path, 'rb') as f:
                agent_dict = pickle.load(f)
            if scenario == 'curling':
                self.agents[scenario] = CurlingAgent(False, net_cls=CurlingNet, net_conf={})
            elif scenario == 'football' or scenario == 'table-hockey':
                self.agents[scenario] = OldGeneralAgent(False, net_cls=OldGeneralNet, net_conf={})
            else:
                self.agents[scenario] = NewGeneralAgent(False, net_cls=NewGeneralNet, net_conf={})
            self.agents[scenario].loads(agent_dict)

        self.curr_agent = None
        self.curr_scenario = None
        self.curr_translator = None
        self.old_general_translator = OldGeneralTranslator()
        self.new_general_translator = NewGeneralTranslator()
        self.curling_translator = CurlingTranslator()

    def get_action(self, obs):
        if obs['game_mode'] == 'NEW GAME':
            md5 = hashlib.md5(obs['agent_obs'].tobytes()).digest()
            for key, scenario in POSSIBLE_INIT_MD5.items():
                if key == md5:
                    self.curr_agent = self.agents[scenario]
                    if scenario == 'curling':
                        self.curr_translator = self.curling_translator
                    elif scenario == 'football' or scenario == 'table-hockey':
                        self.curr_translator = self.old_general_translator
                    else:
                        self.curr_translator = self.new_general_translator
                    break
            else:
                raise ValueError('Unseen initial state:', obs)
            self.curr_translator.reset(obs)
            self.curr_scenario = scenario

        state = self.curr_translator.trans_obs(obs)
        if cannot_act(obs):
            action = [0, 0]
        else:
            if state is not None:
                action = self.curr_agent.infer(state)
                action = self.curr_translator.trans_action(action)
            else:
                action = pre_act_list[self.curr_translator.step_since_inner_start]

        return action


ai = MyAI()


def my_controller(observation, *args):
    action = ai.get_action(observation['obs'])
    action = [[act] for act in action]
    return action
