#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Uncover game
# Compared to the standard game: 1. The list of words is constrained, 2. The votes are secret.

# List of available words, taken from the Small World of Words

import numpy as np
from sknetwork.data import load_netset

data = load_netset('swow')
adjacency = data.adjacency  # graph (if needed)
words = data.names  # words

# Functions to complete; your code must run fast (< 1s on a laptop)

def speak(n_players, player, secret_word="", list_words=[], list_players=[], roles=dict()) -> str:
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
        
    Examples
    --------
    > speak(5, 4, "cat", ["milk"], [3])
    > "lion"
    
    > speak(5, 4, "cat", ["milk", "lion", "house", "cheese", "friend"], [3, 4, 2, 1, 5], {2: "U"})
    > "sleep"
    """
    stop = False
    while not stop:
        word = np.random.choice(words)
        if word not in list_words and word != secret_word:
            stop = True
    return word


def vote(n_players, player, secret_word="", list_words=[], list_players=[], roles=dict()) -> int:
    """
    Vote for a player to eliminate at the end of a round.
    The returned player index cannot be yours, nor a player that has already been eliminated (role known).
    
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
        
    Example
    -------
    > vote(5, 4, "cat", ["milk", "lion", "house", "cheese", "friend"], [3, 4, 2, 1, 5])
    > 2
    """
    players = [i for i in range(1, n_players + 1) if i != player and i not in roles]
    player = np.random.choice(players)
    return player


def guess(n_players, player, list_words=[], list_players=[], roles=dict()) -> str:
    """
    You are Mr White and you have just been eliminated.
    Guess the secret word of Civilians.
    
    Parameters
    ----------
    n_players: int
        Number of players.
    player: int
        Your player id (from 1 to n_players).
    list_words: list of string
        List of words given since the start of the game (empty if you start).
    list_players: list of int
        List of players having spoken since the start of the game (empty if you start).
    roles: dict
        Known roles (including yours as Mr White).
        Key = player, Value = role ("C" for Civilian, "U" for Undercover, "W" for Mr White).
        
    Example
    -------
    > guess(5, 1, ["milk", "lion", "house", "cheese", "friend"], [3, 4, 2, 1, 5])
    > "cat"
    """
    word = np.random.choice(words)
    return word