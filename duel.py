#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Uncover game
# Duel against players

# Compared to the standard game: 
# 1. The list of words is constrained.
# 2. The order of players is random at each round.
# 3. The votes are secret.

import numpy as np

# secret words

pairs = []
with open('pairs.csv') as csvfile:
    for row in csvfile.readlines():
        pairs.append(row[:-1].split(','))
        
# players

speak_ = dict()
vote_ = dict()
guess_ = dict()

from random_player import speak, vote, guess
team = "A"
speak_[team] = speak
vote_[team] = vote
guess_[team] = guess

from random_player import speak, vote, guess
team = "B"
speak_[team] = speak
vote_[team] = vote
guess_[team] = guess

from random_player import speak, vote, guess
team = "C"
speak_[team] = speak
vote_[team] = vote
guess_[team] = guess

# single game

def match(roles, teams, order, civilian_word, undercover_word):
    result = []
    
    # secret words
    secret_word = dict()
    for player, role in roles.items():
        if role == "C":
            secret_word[player] = civilian_word
        elif role == "U":
            secret_word[player] = undercover_word
        else:
            secret_word[player] = ""
    
    # game
    n_players = len(roles)
    list_words = []
    list_players = []
    log = []
    known_roles = dict()
    
    while 1:

        # speak
        for player in order:
            team = teams[player]
            try:
                arguments = n_players, player, secret_word[player], list_words, list_players, known_roles
                word = speak_[team](*arguments)
                word = str(word)
            except Exception as e:
                print("Error on speak...", team)
                print(arguments)
                print(e)
            list_words.append(word)
            list_players.append(player)

        # vote
        votes = []
        for player in order:
            team = teams[player]
            votes.append(vote_[team](n_players, player, secret_word[player], list_words, list_players, known_roles))

        players_, counts = np.unique(votes, return_counts=True)
        candidates = [player for player, count in zip(players_, counts) if count == max(counts)]
        player = np.random.choice(candidates)
        known_roles[player] = roles[player]

        # guess
        if roles[player] == "W":
            team = teams[player]
            word = guess_[team](n_players, player, list_words, list_players, known_roles)
            word = str(word)

        if word == civilian_word:
            return "W"
            
        order = list(set(roles) - set(known_roles))
        remaining_roles = {roles[player] for player in order}
        n_civilians = len([player for player in order if roles[player] == "C"])
        
        if remaining_roles == {"C"}:
            return "C"
        if remaining_roles == {"U", "W"} or (remaining_roles == {"U", "W", "C"} and n_civilians == 1):
            return "I"
        if remaining_roles == {"C", "W"} and n_civilians == 1:
            return "W"
        if remaining_roles == {"C", "U"} and n_civilians == 1:
            return "U"

        np.random.shuffle(order)
        
# duel between 2 players

def duel(team_A, team_B, n_players=5, n_undercover=1, n_games=10, seed=0):
    np.random.seed(seed)
    
    roles = ["C", "U", "W"]
    results = {team_A: {role: 0 for role in roles}, team_B: {role: 0 for role in roles}}
    
    for t in range(n_games):
        # init roles
        players = np.arange(n_players) + 1
        roles = {player: "C" for player in players}

        for player in np.random.choice(players, size=n_undercover, replace=False):
            roles[player] = "U"
        player_ = np.random.choice([player for player in roles if roles[player] == "C"])
        roles[player_] = "W"

        # init order
        order = players
        np.random.shuffle(order)

        # init words
        civilian_word, undercover_word = pairs[np.random.choice(len(pairs))]

        # play
        for team, opponent in [(team_A, team_B), (team_B, team_A)]:
            for attribute_role in ["C", "W", "U"]:
                teams = dict()
                for player, role in roles.items():
                    if attribute_role == role:
                        teams[player] = team
                    else:
                        teams[player] = opponent

                try:
                    result = match(roles, teams, order, civilian_word, undercover_word)
                except Exception as e: 
                    print(e)
                    print(roles, teams, order, civilian_word, undercover_word)
                if result == "C":
                    if attribute_role == "C":
                        results[team]["C"] += 1
                    else:
                        results[opponent]["C"] += 1
                elif result == "U":
                    if attribute_role == "U":
                        results[team]["U"] += 1
                    else:
                        results[opponent]["U"] += 1
                elif result == "W":
                    if attribute_role == "W":
                        results[team]["W"] += 1
                    else:
                        results[opponent]["W"] += 1
                elif result == "I":
                    if attribute_role == "C":
                        results[opponent]["W"] += 1
                        results[opponent]["U"] += 1
                    elif attribute_role == "U":
                        results[team]["U"] += 1
                        results[opponent]["W"] += 1
                    elif attribute_role == "W":
                        results[team]["W"] += 1
                        results[opponent]["U"] += 1

    return results
    
    
# all duels

def duels(teams, n_players, n_undercover, n_games=5, seed=0):
    gains = {"C": 2 * (n_players - 1 - n_undercover), "U": 10, "W": 6}
    roles = list(gains)
    
    scores = {team: {role: 0 for role in roles} for team in teams}
    for i in range(len(teams)):
        print(teams[i])
        for j in range(i):
            results = duel(teams[i], teams[j], n_players, n_undercover, n_games, seed)
            for team in results:
                for role in results[team]:
                    scores[team][role] += results[team][role]
                    
    for team in scores:
        score = 0
        for role, gain in gains.items():
            score += gain * scores[team][role]
        scores[team]["Score"] = score
        
    roles = ["C", "U", "W", "Score"]

    print("Victory as C, U, W, Score")
    scores_ = {team: scores[team]["Score"] for team in scores}
    teams = sorted(scores_, key=scores_.get, reverse=True)
    for team in teams:
        results = team + ": " + ",".join([str(scores[team][role]) for role in roles])
        print(results)
        
    return scores

# duels

teams = list(speak_)
duels(teams, 7, 2, 5)
