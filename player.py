#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Uncover game
# Compared to the standard game: 1. The list of words is constrained, 2. The votes are secret.

# List of available words, taken from the Small World of Words

import numpy as np
from sknetwork.data import load_netset

data = load_netset('swow')
adjacency = data.adjacency  # graph
words = data.names  # words

# Number of players 
# We assume 1 Mr White and 1 Uncover

N = 5

# Functions to complete; your code must run fast (< 1s on a laptop)

def speak(player, secret_word="", list_words=[], list_players=[], roles=dict()) -> str:
    """
    Give a word to other players.
    The word must belong to the list of available words.
    It cannot be the secret word, nor a word that has already been given.
    
    Parameters
    ----------
    player: int
        Your player index (from 1 to N).
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
    > speak(4, "cat", ["milk"], [3])
    > "lion"
    
    > speak(4, "cat", ["milk", "lion", "house", "cheese", "friend"], [3, 4, 2, 1, 5], {2: "U"})
    > "sleep"
    """
    return None


def vote(player, secret_word="", list_words=[], list_players=[], roles=dict()) -> int:
    """
    Vote for a player to eliminate at the end of a round.
    The returned player index cannot be yours, nor a player that has already been eliminated (role known).
    
    Parameters
    ----------
    player: int
        Your player index (from 1 to N).
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
    > vote(4, "cat", ["milk", "lion", "house", "cheese", "friend"], [3, 4, 2, 1, 5])
    > 2
    """
    return None


def guess(player, list_words=[], list_players=[], roles=dict()) -> str:
    """
    You are Mr White and you have just been eliminated.
    Guess the secret word of Civilians.
    
    Parameters
    ----------
    player: int
        Your player index (from 1 to N).
    list_words: list of string
        List of words given since the start of the game (empty if you start).
    list_players: list of int
        List of players having spoken since the start of the game (empty if you start).
    roles: dict
        Known roles.
        Key = player, Value = role ("C" for Civilian, "U" for Undercover, "W" for Mr White).
        
    Example
    -------
    > guess(1, ["milk", "lion", "house", "cheese", "friend"], [3, 4, 2, 1, 5])
    > "cat"
    """
    return None