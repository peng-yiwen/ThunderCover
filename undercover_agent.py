import numpy as np
from sknetwork.data import load_netset

# Load SWOW word association graph
data = load_netset("swow")
adjacency = data.adjacency  # sparse adjacency matrix
words = np.array([str(word) for word in data.names])
word2id = {w: i for i, w in enumerate(words)}

# --- Precompute neighbors for speed ---
TOPK = 48  # number of neighbors per word to cache

neighbors = {}
for i, w in enumerate(words):
    row = adjacency[i].tocoo()
    if row.nnz > 0:
        idx = row.col
        weights = row.data # direct neighbors
        order = np.argsort(-weights)[:TOPK]
        neighbors[w] = [(words[j], weights[k]) for j, k in zip(idx[order], order)] # similarity is weights
    else:
        neighbors[w] = []


def _similarity(w1: str, w2: str) -> float:
    if w1 not in neighbors or w2 not in neighbors:
        return 0.0
    sim12 = dict(neighbors[w1]).get(w2, 0.0)
    sim21 = dict(neighbors[w2]).get(w1, 0.0)
    return max(sim12, sim21)


def _topic_score(c: str, spoken: list[str]) -> float:
    if not spoken:
        return 0.0
    return np.mean([_similarity(c, w) for w in spoken])


def speak(n_players, player, secret_word="", list_words=[], list_players=[], roles=dict()) -> str:
    # Case 1: Mr White
    if secret_word == "":
        if not list_words:
            return "thing"
        candidates = set()
        for w in list_words:
            for n, _ in neighbors.get(w, [])[:8]:
                if n not in list_words:
                    candidates.add(n)
        if not candidates:
            candidates = set(words)
        scored = [(c, _topic_score(c, list_words)) for c in candidates]
        scored.sort(key=lambda x: -x[1])
        return scored[0][0]

    # Case 2: Civilian or Undercover
    spoken = list_words
    candidates = set(n for n, _ in neighbors.get(secret_word, []))
    for w in spoken:
        for n, _ in neighbors.get(w, [])[:4]:
            candidates.add(n)
    candidates.discard(secret_word)
    candidates = [c for c in candidates if c not in spoken]
    if not candidates:
        candidates = [w for w in words if w not in spoken]

    topic_scores = {c: _topic_score(c, spoken) for c in candidates}
    fidelity_scores = {c: _similarity(c, secret_word) for c in candidates}

    round_idx = len(spoken) // max(1, n_players)

    scored = []
    for c in candidates:
        B = topic_scores[c]
        F = fidelity_scores[c]
        if round_idx == 0:
            score = 0.3 * F + 0.6 * B
        elif round_idx == 1:
            score = 0.5 * F + 0.4 * B
        else:
            score = 0.6 * F + 0.3 * B
        scored.append((c, score))
        # undercover why more fidelity?

    scored.sort(key=lambda x: -x[1])
    return scored[0][0]


def vote(n_players, player, secret_word="", list_words=[], list_players=[], roles=dict()) -> int:
    # Build player -> words mapping
    player_words = {i: [] for i in range(1, n_players + 1)}
    for w, p in zip(list_words, list_players):
        player_words[p].append(w)

    alive_players = [i for i in range(1, n_players + 1) if i != player and i not in roles]
    if not alive_players:
        return (player % n_players) + 1  # fallback

    all_spoken = list_words

    suspicions = {}
    for p in alive_players:
        words_p = player_words.get(p, [])
        if not words_p:
            suspicions[p] = 0.5
            continue
        cross_sim = np.mean([
            np.mean([_similarity(wp, wq) for wq in all_spoken if wq not in words_p])
            for wp in words_p
        ]) if len(all_spoken) > len(words_p) else 0
        consistency = np.mean([
            _similarity(w1, w2) for i, w1 in enumerate(words_p) for j, w2 in enumerate(words_p) if i < j
        ]) if len(words_p) > 1 else _topic_score(words_p[0], all_spoken)
        max_topic = max(_topic_score(w, all_spoken) for w in words_p)
        suspicion = (1 - cross_sim) * 0.5 + (1 - consistency) * 0.2 + (1 - max_topic) * 0.2
        suspicions[p] = suspicion

    target = max(suspicions.items(), key=lambda x: x[1])[0]
    return target


def guess(n_players, player, list_words=[], list_players=[], roles=dict()) -> str:
    # popular words to start
    if not list_words:
        return "life"

    # Candidate pool: neighbors of spoken words + spoken words
    candidates = set(list_words)
    for w in list_words:
        for n, _ in neighbors.get(w, [])[:TOPK]:
            candidates.add(n)

    # Score candidates
    scored = []
    for c in candidates:
        clue_sim = np.mean([_similarity(c, w) for w in list_words])
        centrality = _topic_score(c, list_words)
        hub_penalty = np.log1p(len(neighbors.get(c, []))) / 10.0
        score = 0.8 * clue_sim + 0.2 * centrality - hub_penalty
        scored.append((c, score))

    scored.sort(key=lambda x: -x[1])
    for cand, _ in scored:
        if cand not in list_words:
            return cand

    return scored[0][0] if scored else "thing"
