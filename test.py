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

data = load_netset('swow', verbose=False)
adjacency = data.adjacency  # graph (if needed)
words = [str(word) for word in data.names]  # words

# This is a function to test your player

import time
# from player import speak, vote, guess
from undercover_agent import speak, vote, guess

def test_player(roles, rounds):
    # init
    secret = np.random.choice(words, size=2, replace=False)
    civilian_word, undercover_word = str(secret[0]), str(secret[1])
    secret_word = dict()
    for player, role in roles.items():
        if role == "C":
            secret_word[player] = civilian_word
        elif role == "U":
            secret_word[player] = undercover_word
        else:
            secret_word[player] = ""
    
    # test
    players = list(roles.keys())
    n_players = len(players)
    list_words = [str(word) for word in np.random.choice(words, size=len(rounds) * n_players, replace=False)]
    list_players = []
    known_roles = dict()
    for player_ in rounds:
        np.random.shuffle(players)
        # speak
        for player in players:
            try: 
                t = len(list_players)
                arguments = [n_players, player, secret_word[player], list_words[:t], list_players, known_roles] 
                t0 = time.time()
                word = speak(*arguments)
                t1 = time.time()
                
            except: 
                print(f"Error on speak...\n Arguments = {arguments}")
            if word not in words:
                print(f"Error on speak...\n Incorrect word {word}\n Arguments = {arguments}")
            if t1 - t0 > 0.1:
                print(f"Speak too slow... \n Time = {t1 - t0} \n Arguments = {arguments}")
            list_players.append(player)
        # vote
        for player in players:
            try: 
                t = len(list_players)
                t0 = time.time()
                candidate = vote(n_players, player, secret_word[player], list_words[:t], list_players, known_roles)
                t1 = time.time()
                arguments = [n_players, player, secret_word[player], list_words[:t], list_players, known_roles]
            except: 
                print(f"Error on vote... \n Arguments = {arguments}")
            if candidate not in players or candidate == player:
                print(f"Error on vote... \n Incorrect player {candidate} \n Arguments = {arguments}")        
            if t1 - t0 > 0.1:
                print(f"Vote too slow... \n Time = {t1 - t0} \n Arguments = {arguments}")        
        known_roles[player_] = roles[player_]
        players = list(set(roles) - set(known_roles))

    # guess 
    try: 
        player = rounds[-1]
        t0 = time.time()
        word = guess(n_players, player, list_words[:t], list_players, known_roles)
        t1 = time.time()
        arguments = [n_players, player, list_words[:t], list_players, known_roles]
    except: 
        print(f"Error on guess...\n Arguments = {arguments}")   
    if t1 - t0 > 0.1:
        print(f"Guess too slow... \n Time = {t1 - t0} \n Arguments = {arguments}")     
        
        
print("If no message appears, your player passes the test. Otherwise, please modify your code.")

test_player({1: "C", 2: "W", 3:"C", 4: "C", 5: "U"}, [3, 4, 2])
test_player({1: "U", 2: "C", 3:"C", 4: "C", 5: "U", 6: "C", 7: "W"}, [1, 6, 5, 2, 7])
