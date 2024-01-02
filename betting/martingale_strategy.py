import random
import numpy as np

 # Number of players simulated
PLAYERS = 1000
# Chance of winning a single bet
WIN_CHANCE = 18/37
# Fee payed when the player win: if the number on the roulette is black, 
# the casino pays an amount equal to the bet, so you have 2 times the cash you bet
WIN_FEE = 2
# Quota remaining ito the player when he lose: at the casino you lose all your bet
# so the multiplier is 0
LOSE_FEE = 0

# Player data
STARTUP_CAPITAL = 1000
# Player strategy
# The inital bet and the multiplier incase of lost bet
BASE_BET = 1
BET_MULTIPLIER = 2

def run_player(target_capital):
    '''
    Run the simulation for a single player. Return the full history, bet by bet, and some summaries.
    '''
    # At the end of the simulation, this will be True if the target capital was reached
    # False if the player lost all capital
    win = None
    final_capital = None
    capital = STARTUP_CAPITAL
    sequence_count = 0
    bet_count = 0
    history = [{'capital': capital, 'sequence': sequence_count, 'bet': bet_count}]

    # The game will stop when the player run out of capital or reach the target capital
    stop_game = False
    while not stop_game:

        # A new betting sequence starts
        sequence_count += 1
        bet = BASE_BET
        stop_sequence = False

        while not stop_sequence:
            # The player does another bet
            bet_count +=  1
            capital -= bet

            if random.random() < WIN_CHANCE:
                # The player wins, receives the cash and the bet sequence stops
                capital += (bet * WIN_FEE)
                stop_sequence = True
                # If the player reaches the target capital, the entire game stops
                if capital >= target_capital:
                    win = True
                    final_capital = capital
                    stop_game = True
            else:
                # The player loses, receives the eventual fee and continues the game if it has enough capital
                capital += (bet * LOSE_FEE)
                if capital >= (bet * BET_MULTIPLIER):
                    # Enough capital: the game continues
                    bet = bet * BET_MULTIPLIER
                else:
                    # Not enough capital, this ends the sequence
                    stop_sequence = True
                if capital < BASE_BET:
                    # No more capital, even to restart a sequence, that's the end.
                    win = False
                    final_capital = capital
                    stop_game = True
            # Update history
            history += [{'capital': capital, 'sequence': sequence_count, 'bet': bet_count}]
    return win, final_capital, history

def run_simulation(total_players, target_capital):
    players = []
    for player in range(0, total_players):
        win, final_capital, history = run_player(target_capital)
        players += [
            {'player': player, 'win': win, 'final_capital': final_capital, 'history': history}
        ]
    return players