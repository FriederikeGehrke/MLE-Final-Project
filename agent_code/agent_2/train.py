"""
Training for mlp_agent: same online Q-learning update as simple_agent, same rewards,
but the TD error is backpropagated through a one-hidden-layer network instead of
being applied directly to a linear feature vector.

    delta = r + GAMMA * max_a' Q(s',a') - Q(s,a)
    params <- params + LR * delta * d/d(params) Q(s,a)
"""
import csv
import os
import shutil

import numpy as np

import events as e
import settings as s
from agent_code.agent_1.callbacks import ACTIONS, F, GAMMA, action_features, allowed
from .callbacks import WEIGHTS_FILE, forward

LEARNING_RATE = 0.005     # lower than simple_agent's 0.01 -- backprop through tanh is less stable
EPSILON_START, EPSILON_END, EPSILON_ROUNDS = 0.2, 0.02, 2000

# identical reward scheme to simple_agent/train.py, so the two models are compared
# under the same objective, not just the same features
R_COIN, R_DEATH, R_CRATE, R_TRAP, R_BOMB = 1.0, -5.0, 0.2, 5.0, -0.1

LOG_FILE = 'training_log.csv'
LOG_COLUMNS = ['round', 'score', 'coins', 'kills', 'crates', 'survived', 'steps', 'epsilon']


def setup_training(self):
    self.round_stats = dict.fromkeys(['coins', 'kills', 'crates', 'survived'], 0)
    if os.path.isfile(LOG_FILE):
        with open(LOG_FILE) as file:
            self.rounds_played = sum(1 for _ in file) - 1
    else:
        self.rounds_played = 0
        with open(LOG_FILE, 'w', newline='') as file:
            csv.writer(file).writerow(LOG_COLUMNS)
    update_epsilon(self)


def update_epsilon(self):
    frac = min(1.0, self.rounds_played / EPSILON_ROUNDS)
    self.epsilon = EPSILON_START + frac * (EPSILON_END - EPSILON_START)


def reward(events, phi):
    r = R_COIN * events.count(e.COIN_COLLECTED)
    if e.GOT_KILLED in events:
        r += R_DEATH
    if e.BOMB_DROPPED in events:
        r += R_BOMB + R_CRATE * phi[F['bomb_crates']] + R_TRAP * phi[F['bomb_traps']]
    return r


def backprop_step(params, phi, target, lr):
    """One gradient step of (target - Q(phi))^2 / 2 w.r.t. all four parameters."""
    q, h = forward(params, phi)
    err = target - q
    params['W2'] += lr * err * h
    params['b2'] += lr * err
    dz1 = err * params['W2'] * (1 - h ** 2)     # backprop through tanh
    params['W1'] += lr * np.outer(dz1, phi)
    params['b1'] += lr * dz1


def td_update(self, old_state, action, events, new_state):
    phi = action_features(old_state)[0][ACTIONS.index(action)]
    target = reward(events, phi)
    if new_state is not None:
        feats, valid, doomed = action_features(new_state)
        q_next, _ = forward(self.params, feats[allowed(valid, doomed)])
        target += GAMMA * q_next.max()
    backprop_step(self.params, phi, target, LEARNING_RATE)


def count(self, events):
    self.round_stats['coins'] += events.count(e.COIN_COLLECTED)
    self.round_stats['kills'] += events.count(e.KILLED_OPPONENT)
    self.round_stats['crates'] += events.count(e.CRATE_DESTROYED)
    self.round_stats['survived'] += events.count(e.SURVIVED_ROUND)


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    td_update(self, old_game_state, self_action, events, new_game_state)
    count(self, events)


def end_of_round(self, last_game_state, last_action, events):
    td_update(self, last_game_state, last_action, events, None)
    count(self, events)
    st = self.round_stats
    self.rounds_played += 1
    with open(LOG_FILE, 'a', newline='') as file:
        csv.writer(file).writerow([
            self.rounds_played, s.REWARD_COIN * st['coins'] + s.REWARD_KILL * st['kills'],
            st['coins'], st['kills'], st['crates'], st['survived'],
            last_game_state['step'], round(self.epsilon, 4)])
    self.round_stats = dict.fromkeys(st, 0)
    update_epsilon(self)
    np.savez(WEIGHTS_FILE, **self.params)
    if self.rounds_played % 100 == 0:
        shutil.copy(WEIGHTS_FILE, f'weights_round_{self.rounds_played}.npz')