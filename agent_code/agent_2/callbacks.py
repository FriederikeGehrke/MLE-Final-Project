"""
Second model: a small neural network on top of the SAME features and the SAME safety
shield as simple_agent. The only thing that changes is the function approximator:

    simple_agent:  Q(s,a) = w . phi(s,a)                (linear)
    mlp_agent:     Q(s,a) = W2 . tanh(W1 . phi(s,a) + b1) + b2   (one hidden layer)

Reusing action_features() means both agents see exactly the same information and the
same "doomed"/"contested" safety logic, so any performance difference is due to the
function class, not to different inputs.
"""
import os
import random

import numpy as np

from agent_code.agent_1.callbacks import ACTIONS, FEATURES, action_features, allowed

HIDDEN = 16
WEIGHTS_FILE = 'weights.npz'


def init_params(rng):
    n_in = len(FEATURES)
    return {
        'W1': rng.normal(0, 0.3, (HIDDEN, n_in)),
        'b1': np.zeros(HIDDEN),
        'W2': rng.normal(0, 0.3, HIDDEN),
        'b2': 0.0,
    }


def forward(params, phi):
    """phi: (..., n_features) -> (q, hidden_activation)."""
    z1 = phi @ params['W1'].T + params['b1']
    h = np.tanh(z1)
    q = h @ params['W2'] + params['b2']
    return q, h


def setup(self):
    rng = np.random.default_rng(0)
    self.params = init_params(rng)
    if os.path.isfile(WEIGHTS_FILE):
        saved = np.load(WEIGHTS_FILE)
        self.params = {k: saved[k] for k in self.params}
    elif not self.train:
        self.logger.warning('No trained weights found -- playing with random init.')


def act(self, game_state):
    feats, valid, doomed = action_features(game_state)
    idx = np.flatnonzero(allowed(valid, doomed))
    if self.train and random.random() < self.epsilon:
        return ACTIONS[random.choice(idx)]
    q, _ = forward(self.params, feats[idx])
    best = idx[q >= q.max() - 1e-9]
    return ACTIONS[random.choice(best)]