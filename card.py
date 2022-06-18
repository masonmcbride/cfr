from enum import Enum
"""
- This file represents the deck used in the kuhn_cfr file
- You can change the amount of cards, give higher cards higher int value
- so comparisons work in the payoff eval 
"""


class Card(Enum):
    K = 13
    Q = 12
    J = 11
#    T = 10
#    N = 9