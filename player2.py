import numpy as np
from sknetwork.data import load_netset

# Load SWOW word association graph
data = load_netset("swow")
adjacency = data.adjacency  # sparse adjacency matrix
words = np.array([str(word) for word in data.names])
word2id = {w: i for i, w in enumerate(words)}

# --- Precompute neighbors for speed ---
TOPK = 64  # number of neighbors per word to cache

neighbors = {}
for i, w in enumerate(words):
    row = adjacency[i].tocoo()
    if row.nnz > 0:
        idx = row.col
        weights = row.data
        order = np.argsort(-weights)[:TOPK]
        neighbors[w] = [(words[j], weights[k]) for j, k in zip(idx[order], order)]
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


def _compute_p_majority(secret_word: str, spoken: list[str], bg_nei: int = 50, max_bg: int = 1000) -> float:
    """Estimate probability that your secret_word belongs to the majority (civilian) group.

    Uses a background-percentile approach: compares mean similarity of secret_word to spoken words
    against the distribution of mean-similarities for a background set built from neighbors of spoken words.
    Returns a value in [0,1].
    """
    if not spoken:
        return 0.5

    # raw cross similarity: mean similarity between secret and spoken words
    cross_sim = np.mean([_similarity(secret_word, w) for w in spoken])

    # build background pool from neighbors of spoken words
    BG = set()
    for w in spoken:
        for n, _ in neighbors.get(w, [])[:bg_nei]:
            BG.add(n)
    # fallback to a small random sample from vocab if BG empty or too small
    if not BG:
        # sample up to max_bg words from the full vocabulary
        rng = np.random.default_rng()
        sampled = rng.choice(words, size=min(len(words), max_bg), replace=False)
        BG = set(sampled)
    else:
        # if BG larger than max_bg, sample
        if len(BG) > max_bg:
            BG = set(np.random.choice(list(BG), size=max_bg, replace=False))

    # compute bg_sims: for each b in BG, mean similarity to spoken
    bg_sims = []
    for b in BG:
        bg_sims.append(np.mean([_similarity(b, w) for w in spoken]))

    if not bg_sims:
        return 0.5

    bg_sims = np.array(bg_sims)
    pct = float((bg_sims <= cross_sim).sum()) / len(bg_sims)
    return pct


def speak(n_players, player, secret_word="", list_words=[], list_players=[], roles=dict()) -> str:
    """Choose a word to speak.

    New behavior:
    - computes a percentile-based cross-similarity p_majority
    - interpolates fidelity vs topic-blending weights using p_majority
    - scores candidates and samples among top-K with temperature depending on p_majority
    """
    # Mr White behavior unchanged in spirit: pick a bridge word to blend
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

    spoken = list_words

    # --- compute p_majority (probability your secret_word is aligned with the room) ---
    p_majority = _compute_p_majority(secret_word, spoken)

    # weight interpolation for fidelity
    wF_min, wF_max = 0.20, 0.75
    weight_fidelity = wF_min + p_majority * (wF_max - wF_min)
    weight_topic = 1.0 - weight_fidelity

    # round-based multiplier (still modest influence)
    round_idx = len(spoken) // max(1, n_players)
    round_multiplier = 1.0 + 0.15 * min(round_idx, 3)
    # if likely majority, slightly increase fidelity more over rounds
    if p_majority > 0.6:
        weight_fidelity = min(0.95, weight_fidelity * round_multiplier)
        weight_topic = 1.0 - weight_fidelity

    # --- candidate generation ---
    candidates = set(n for n, _ in neighbors.get(secret_word, []))
    for w in spoken:
        for n, _ in neighbors.get(w, [])[:4]:
            candidates.add(n)
    candidates.discard(secret_word)
    candidates = [c for c in candidates if c not in spoken]
    if not candidates:
        candidates = [w for w in words if w not in spoken]

    # Precompute scores
    topic_scores = {c: _topic_score(c, spoken) for c in candidates}
    fidelity_scores = {c: _similarity(c, secret_word) for c in candidates}

    scored = []
    # penalties
    top_secret_neis = [n for n, _ in neighbors.get(secret_word, [])[:3]]
    for c in candidates:
        B = topic_scores[c]
        F = fidelity_scores[c]
        # base blended score
        score = weight_fidelity * F + weight_topic * B
        # penalize if c is an overly obvious synonym (top-3 neighbor)
        if c in top_secret_neis:
            score -= 0.30
        # weak hub penalty: average neighbor weight normalized
        neigh_weights = [w for _, w in neighbors.get(c, [])]
        hubness = float(np.mean(neigh_weights)) if neigh_weights else 0.0
        score -= 0.10 * hubness
        scored.append((c, score))

    # choose among top-K with temperature depending on p_majority
    scored.sort(key=lambda x: -x[1])
    TOP_SAMPLE = min(3, len(scored))
    topk = scored[:TOP_SAMPLE]
    if not topk:
        # fallback
        return candidates[0]

    scores_arr = np.array([s for _, s in topk], dtype=float)
    # temperature: lower when confident majority (decisive), higher when likely undercover
    T = 0.35 + 0.85 * (1.0 - p_majority)  # in [0.35, 1.2]
    # softmax sampling
    max_s = np.max(scores_arr)
    prefs = np.exp((scores_arr - max_s) / T)
    probs = prefs / prefs.sum()
    choice_idx = np.random.choice(len(topk), p=probs)
    return topk[choice_idx][0]


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
