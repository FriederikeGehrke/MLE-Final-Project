# Agent with linear Q-learning on hand-made features

# For every possible action a feature vector (how close are coins / crates / opponents afterwards, is the action safe, ...) is computed. 
# The Q-value of an action is feature vector @ weights. Only the weights are learned (train.py).


import json
import os
import random
from collections import deque

import numpy as np

import settings as s

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
STEP = {'UP': (0, -1), 'RIGHT': (1, 0), 'DOWN': (0, 1), 'LEFT': (-1, 0)}

FEATURES = [
    'coin_prox',       # GAMMA^(path length to the nearest coin)
    'crate_prox',      # GAMMA^(path length to the nearest tile next to a crate)
    'opponent_prox',   # GAMMA^(path length to the nearest opponent)
    'bomb_crates',     # BOMB only: number of crates the blast would destroy
    'bomb_traps',      # BOMB only: opponents in the blast that can't escape it
    'bomb',
    'wait',
    'doomed',          # if after the action death is unavoidable
    'contested',       # not doomed, but doomed if tiles near opponents are blocked
]
F = {name: i for i, name in enumerate(FEATURES)}

GAMMA = 0.9  # discount of Q-learning (also the base of _prox features)
SHIELD = True    # don't pick an action that makes death unavoidable (unless all actions do)
HORIZON = s.BOMB_TIMER + s.EXPLOSION_TIMER + 1   # after this number of steps all known explosions are over
CONTEST_RADIUS = 2  # assume opponents block every tile they can reach in this number of steps
FAR = 99  # distance for unreachable tiles
WEIGHTS_FILE = 'weights.json'


def blast(field, pos):
    """ Tiles hit by a bomb (stopped by stone walls, not by crates) """
    x, y = pos
    tiles = [pos]
    for dx, dy in STEP.values():
        for i in range(1, s.BOMB_POWER + 1):
            t = (x + dx * i, y + dy * i)
            if field[t] == -1:
                break
            tiles.append(t)
    return tiles


def mark_blast(lethal, tiles, timer):
    """ 
    lethal[t, x, y]: standing on (x, y) after t more actions is deadly 
    A bomb with countdown `timer` is deadly at t = timer+1 and timer+2 
    """
    xs, ys = zip(*tiles)
    lethal[timer + 1: timer + 1 + s.EXPLOSION_TIMER, xs, ys] = True


def survives(lethal, free, start, t):
    """
    Does the agent stays alive from time t until all explosions are over?
    Search over (tile, time): each step stay or move to a free, non-lethal neighbour
    `alive` is the set of tiles it could be standing on
    """
    alive = {start}
    while t < HORIZON:
        t += 1
        nxt = set()
        for (x, y) in alive:
            for q in ((x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if not lethal[t][q] and (q == (x, y) or free[q]):
                    nxt.add(q)
        if not nxt:
            return False
        alive = nxt
    return True


def distances(passable, sources):
    """ Path length from every tile to the nearest source (multi-source BFS: start with all sources at once) """
    dist = np.full(passable.shape, FAR)
    queue = deque()
    for p in sources:
        dist[p] = 0
        queue.append(p)
    while queue:
        x, y = queue.popleft()
        for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if passable[q] and dist[q] == FAR:
                dist[q] = dist[x, y] + 1
                queue.append(q)
    return dist


def prox(dist):
    return 0.0 if dist >= FAR else GAMMA ** dist # 1 when standing on the target, smaller the further away, 0 if unreachable


def action_features(gs):
    """Returns (feats, valid, doomed) for all actions."""
    field, bombs, expl = gs['field'], gs['bombs'], gs['explosion_map']
    _, _, can_bomb, pos = gs['self']
    opps = [o[3] for o in gs['others']]

    # passable (no walls, crates and bombs block)
    passable = (field == 0)  
    for b, _ in bombs:
        passable[b] = False
    # free (what next steps are actually possible) - passable & no opponent
    free = passable.copy() 
    for o in opps:
        free[o] = False

    # where and when is it deadly
    lethal = np.zeros((HORIZON + 1,) + field.shape, bool)
    for t in range(1, int(expl.max()) + 1):  # explosions that are already burning
        lethal[t] |= expl >= t
    for b, timer in bombs:  # bombs that will explode
        mark_blast(lethal, blast(field, b), timer)

    coin_dist = distances(passable, gs['coins'])
    # bombing spots: free tiles that touch at least one crate
    spots = {(int(x) + dx, int(y) + dy)
             for x, y in zip(*np.nonzero(field == 1)) for dx, dy in STEP.values()
             if passable[int(x) + dx, int(y) + dy]}
    crate_dist = distances(passable, spots)
    opp_dist = distances(passable, opps)

    feats = np.zeros((len(ACTIONS), len(FEATURES)))
    valid = np.zeros(len(ACTIONS), bool)
    doomed = np.zeros(len(ACTIONS), bool)

    for i, a in enumerate(ACTIONS):
        # position after action (skip actions that aren't possible)
        if a in STEP:
            new = (pos[0] + STEP[a][0], pos[1] + STEP[a][1])
            if not free[new]:
                continue
        elif a == 'BOMB' and not can_bomb:
            continue
        else:
            new = pos
        valid[i] = True
        f = feats[i]
        f[F['coin_prox']] = prox(coin_dist[new])
        f[F['crate_prox']] = prox(crate_dist[new])
        f[F['opponent_prox']] = prox(opp_dist[new])
        f[F['wait']] = (a == 'WAIT')

        # for bombs
        lethal_a, free_a = lethal, free
        if a == 'BOMB':
            tiles = blast(field, pos)
            lethal_a = lethal.copy()
            mark_blast(lethal_a, tiles, s.BOMB_TIMER)
            free_a = free.copy()
            free_a[pos] = False   # the bomb blocks its own tile once I've left
            passable_a = passable.copy()
            passable_a[pos] = False
            f[F['bomb']] = 1
            f[F['bomb_crates']] = sum(field[t] == 1 for t in tiles)
            # an opponent is trapped if it stands in the blast and has no escape (ignoring other agents)
            f[F['bomb_traps']] = sum(o in tiles and not survives(lethal_a, passable_a, o, 0)
                                     for o in opps)
        doomed[i] = lethal_a[1][new] or not survives(lethal_a, free_a, new, 1)
        f[F['doomed']] = doomed[i]
        # contested: safe in theory, but not if opponents get in the way of my escape route
        if not doomed[i]:
            f[F['contested']] = not survives(lethal_a, free_a & (opp_dist > CONTEST_RADIUS), new, 1)
    return feats, valid, doomed


def allowed(valid, doomed):
    ok = valid & ~doomed if SHIELD else valid # shield: only actions that don't mean certain death
    return ok if ok.any() else valid


def setup(self):
    self.weights = np.zeros(len(FEATURES))
    if os.path.isfile(WEIGHTS_FILE):
        with open(WEIGHTS_FILE) as file:
            saved = json.load(file)
        self.weights = np.array([saved.get(name, 0.0) for name in FEATURES])
    elif not self.train:
        self.logger.warning('No trained weights found.')


def act(self, game_state):
    feats, valid, doomed = action_features(game_state)
    idx = np.flatnonzero(allowed(valid, doomed))
    # epsilon-greedy: sometimes a random action while training
    if self.train and random.random() < self.epsilon:
        return ACTIONS[random.choice(idx)]
    q = feats[idx] @ self.weights
    best = idx[q >= q.max() - 1e-9] # random tie-break
    return ACTIONS[random.choice(best)]

