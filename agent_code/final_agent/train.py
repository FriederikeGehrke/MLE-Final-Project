# Training for the linear Q-learning agent 
# Q(s,a) = features(s,a) @ weights, updated after every step with a temporal difference update


import csv
import json
import os
import shutil

import events as e
import settings as s
from .callbacks import ACTIONS, FEATURES, F, GAMMA, WEIGHTS_FILE, action_features, allowed

LEARNING_RATE = 0.003 # in first 700 rounds 0.01, then 0.03 accidently
EPSILON_START, EPSILON_END, EPSILON_ROUNDS = 0.02, 0.02, 2000   # in first 700 rounds linear decay over rounds with 0.1, 0.02, 2000

# Rewards
R_COIN = 1.0     # collected a coin
R_DEATH = -5.0   # died from own bomb or opponent's bomb
R_CRATE = 0.2    # per crate a dropped bomb will destroy  (paid at the moment of dropping)
R_TRAP = 5.0     # per opponent that can't escape a dropped bomb (paid at the moment of dropping, earlier than game reports it)
R_BOMB = -0.1    # cost of every bomb

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
    # linear decay of epsilon
    frac = min(1.0, self.rounds_played / EPSILON_ROUNDS)
    self.epsilon = EPSILON_START + frac * (EPSILON_END - EPSILON_START)


def reward(events, action, phi):
    # phi: features of the action that was taken
    r = R_COIN * events.count(e.COIN_COLLECTED)
    if e.GOT_KILLED in events: # also from own bomb
        r += R_DEATH
    if e.BOMB_DROPPED in events:
        r += R_BOMB + R_CRATE * phi[F['bomb_crates']] + R_TRAP * phi[F['bomb_traps']]
    return r


def td_update(self, old_state, action, events, new_state):
    # new_state is None at the end of a round -> no bootstrapping
    phi = action_features(old_state)[0][ACTIONS.index(action)]
    target = reward(events, action, phi)
    if new_state is not None:
        feats, valid, doomed = action_features(new_state)
        target += GAMMA * (feats[allowed(valid, doomed)] @ self.weights).max()
    self.weights += LEARNING_RATE * (target - phi @ self.weights) * phi


def count(self, events):
    # per-round statistics 
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
    with open(WEIGHTS_FILE, 'w') as file:
        json.dump(dict(zip(FEATURES, self.weights.tolist())), file, indent=2)
    # keep a copy every 200 rounds
    if self.rounds_played % 200 == 0:
        shutil.copy(WEIGHTS_FILE, f'weights_round_{self.rounds_played}.json')