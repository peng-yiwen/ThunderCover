#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Uncover game

# Compared to the standard game:
# 1. The list of words is constrained.
# 2. The order of players is random at each round.
# 3. The votes are secret.

# List of available words, taken from the Small World of Words

import numpy as np
from sknetwork.data import load_netset
from functools import lru_cache

data = load_netset("swow")
adjacency = data.adjacency  # graph (if needed)
words = [str(word) for word in data.names]  # words

# Functions to complete; your code must run fast (less than 100ms on a laptop)


TOPK = 64
# Functions to complete; your code must run fast (less than 100ms on a laptop)


neighbors = {}

for i, w in enumerate(words):
    row = adjacency[i].tocoo()
    if row.nnz > 0:
        idx = row.col
        weights = row.data
        order = np.argsort(-weights)[:TOPK]
        neighbors[w] = {words[idx[j]]: weights[j] for j in order}
    else:
        neighbors[w] = {}


@lru_cache(maxsize=None)
def _similarity(w1: str, w2: str) -> float:
    sim12 = neighbors.get(w1, {}).get(w2, 0.0)
    sim21 = neighbors.get(w2, {}).get(w1, 0.0)
    return max(sim12, sim21)


def _topic_score(c: str, spoken: list[str]) -> float:
    if not spoken:
        return 0.0

    return np.mean([_similarity(c, w) for w in spoken])


def _cross_similarity(secret_word: str, spoken: list[str], bg_nei: int = 50, max_bg: int = 1000):
    """
    Return (cross_sim, pct)
      - cross_sim = mean similarity(secret_word, w) over spoken
      - pct = percentile of cross_sim vs background similarities (BG built from neighbors of spoken words)
    Both are fast and use only the cached neighbor dicts.
    """
    if not spoken:
        return 0.0, 0.5  # no signal -> neutral percentile

    # raw cross similarity
    cross_sim = float(np.mean([_similarity(secret_word, w) for w in spoken]))

    # build background pool from neighbors of spoken words
    BG = set()
    for w in spoken:
        for n in list(neighbors.get(w, {}).keys())[:bg_nei]:
            BG.add(n)

    # fallback: if BG empty sample from whole vocab (small sample)
    if not BG:
        rng = np.random.default_rng()
        sample_size = min(len(words), max_bg)
        sampled = rng.choice(words, size=sample_size, replace=False)
        BG = set(sampled)
    else:
        if len(BG) > max_bg:
            BG = set(np.random.choice(list(BG), size=max_bg, replace=False))

    # compute bg sims (mean similarity to the spoken set)
    bg_sims = []
    for b in BG:
        bg_sims.append(float(np.mean([_similarity(b, w) for w in spoken])))

    if not bg_sims:
        return cross_sim, 0.5

    bg_sims = np.array(bg_sims)
    pct = float((bg_sims <= cross_sim).sum()) / len(bg_sims)

    return cross_sim, pct


def speak(
    n_players, player, secret_word="", list_words=[], list_players=[], roles=dict()
) -> str:
    """
    Give a word to other players.
    The word must belong to the list of available words.
    It cannot be the secret word, nor a word that has already been given.

    Parameters
    ----------
    n_players: int
        Number of players.
    player: int
        Your player id (from 1 to n_players).
    secret_word: string
        Your secret word (empty string if Mr White).
    list_words: list of string
        List of words given since the start of the game (empty if you start).
    list_players: list of int
        List of players having spoken since the start of the game (empty if you start).
    roles: dict
        Known roles.
        Key = player, Value = role ("C" for Civilian, "U" for Undercover, "W" for Mr White).
    """

    # Mr White case: handled near the end (keeps original logic)
    spoken = list_words

    # Candidate pool: neighbors of secret_word + neighbors of recent spoken words
    candidates = set()
    if secret_word:
        candidates.update(neighbors.get(secret_word, {}).keys())
    for w in spoken:
        for n in list(neighbors.get(w, {}).keys())[:4]:
            candidates.add(n)

    # cleanup
    if secret_word:
        candidates.discard(secret_word)
    candidates = [c for c in candidates if c not in spoken]
    if not candidates:
        candidates = [w for w in words if w not in spoken]

    # topic & fidelity
    topic_scores = {c: _topic_score(c, spoken) for c in candidates}
    fidelity_scores = {c: _similarity(c, secret_word) if secret_word else 0.0 for c in candidates}

    # compute cross-similarity percentile -> p_majority
    if secret_word:
        _, p_majority = _cross_similarity(secret_word, spoken)
    else:
        p_majority = 0.5

    # interpolate fidelity weight using p_majority
    wF_min, wF_max = 0.20, 0.75
    weight_fidelity = wF_min + p_majority * (wF_max - wF_min)
    weight_topic = 1.0 - weight_fidelity

    # mild round influence (keeps some of your previous behavior)
    round_idx = len(spoken) // max(1, n_players)
    if p_majority > 0.6:
        round_multiplier = 1.0 + 0.15 * min(round_idx, 3)
        weight_fidelity = min(0.95, weight_fidelity * round_multiplier)
        weight_topic = 1.0 - weight_fidelity

    scored = []
    # penalize overly-obvious neighbors and hubs
    top_secret_neis = []
    if secret_word:
        top_secret_neis = list(neighbors.get(secret_word, {}).keys())[:3]
    for c in candidates:
        B = topic_scores[c]
        F = fidelity_scores[c]
        score = weight_fidelity * F + weight_topic * B
        if c in top_secret_neis:
            score -= 0.30
        neigh_weights = list(neighbors.get(c, {}).values())
        hubness = float(np.mean(neigh_weights)) if neigh_weights else 0.0
        score -= 0.10 * hubness
        scored.append((c, score))

    scored.sort(key=lambda x: -x[1])

    # If Mr White (secret_word empty) fallback selection to blend with room
    if secret_word == "":
        if not list_words:
            return "water" # the most popular word based on PageRank
        candidates = set()
        for w in list_words:
            for n in list(neighbors.get(w, {}).keys())[:8]:
                if n not in list_words:
                    candidates.add(n)
        if not candidates:
            candidates = set(words)
        scored = [(c, _topic_score(c, list_words)) for c in candidates]
        scored.sort(key=lambda x: -x[1])

    return scored[0][0]


def vote(
    n_players, player, secret_word="", list_words=[], list_players=[], roles=dict()
) -> int:
    """
    Vote for a player to eliminate at the end of a round.
    The returned player index cannot be yours, nor a player that has already been eliminated (role known).
    """
    player_words = {i: [] for i in range(1, n_players + 1)}
    for w, p in zip(list_words, list_players):
        player_words[p].append(w)

    alive_players = [
        i for i in range(1, n_players + 1) if i != player and i not in roles
    ]
    if not alive_players:
        return (player % n_players) + 1  # fallback

    all_spoken = list_words

    suspicions = {}
    for p in alive_players:
        words_p = player_words.get(p, [])
        if not words_p:
            suspicions[p] = 0.5
            continue
        cross_sim = (
            np.mean(
                [
                    np.mean(
                        [_similarity(wp, wq) for wq in all_spoken if wq not in words_p]
                    )
                    for wp in words_p
                ]
            )
            if len(all_spoken) > len(words_p)
            else 0
        )
        consistency = (
            np.mean(
                [
                    _similarity(w1, w2)
                    for i, w1 in enumerate(words_p)
                    for j, w2 in enumerate(words_p)
                    if i < j
                ]
            )
            if len(words_p) > 1
            else _topic_score(words_p[0], all_spoken)
        )
        max_topic = max(_topic_score(w, all_spoken) for w in words_p)
        suspicion = (
            (1 - cross_sim) * 0.5 + (1 - consistency) * 0.2 + (1 - max_topic) * 0.2
        )
        suspicions[p] = suspicion

    target = max(suspicions.items(), key=lambda x: x[1])[0]
    return target


def guess(n_players, player, list_words=[], list_players=[], roles=dict()) -> str:
    """
    You are Mr White and you have just been eliminated.
    Guess the secret word of Civilians.
    """
    candidates = set(list_words)
    for w in list_words:
        for n in list(neighbors.get(w, {}).keys())[:TOPK]:
            candidates.add(n)

    scored = []
    for c in candidates:
        clue_sim = np.mean([_similarity(c, w) for w in list_words])
        centrality = _topic_score(c, list_words)
        hub_penalty = np.log1p(len(neighbors.get(c, {}))) / 10.0
        score = 0.8 * clue_sim + 0.2 * centrality - hub_penalty
        scored.append((c, score))

    scored.sort(key=lambda x: -x[1])
    for cand, _ in scored:
        if cand not in list_words:
            return cand

    return scored[0][0] if scored else "cest la vie"
