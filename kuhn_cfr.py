#Monte Carlo Based Counterfactual Regret Optimization with Kuhn Poker
#Players are each dealt 1 card from a deck of 3, actions=[check, bet] 

from typing import List, Dict
import sys
import numpy as np
from card import Card

### (FRIDAY JUNE 17, 2022) Goal: 2-card Kuhn -> 3-card Kuhn
### (Sat. June 18, 2022) In order to change the amount of cards used, edit the card.py folder 

Actions     : List[str] = ['B', 'C'] #bet/call check/fold
NUM_ACTIONS : int  = len(Actions)

class InformationSet:

    __slots__ = ("strategy_sum", "regret_sum", "num_actions")
    
    def __init__(self):
        self.strategy_sum : np.ndarray = np.zeros(NUM_ACTIONS)
        self.regret_sum   : np.ndarray = np.zeros(NUM_ACTIONS)
        self.num_actions  : int = NUM_ACTIONS

    def normalize(self, strategy : np.ndarray) -> np.ndarray:
        if sum(strategy) > 0:
            return strategy/sum(strategy)
        else:
            return np.ones(self.num_actions)/self.num_actions

    def get_strategy(self, transition_prob : float) -> np.ndarray:
        strategy = np.maximum(self.regret_sum, 0)
        strategy = self.normalize(strategy)

        self.strategy_sum += transition_prob * strategy
        return strategy

    def get_average_strategy(self) -> np.ndarray:
        return self.normalize(self.strategy_sum)

class KuhnPoker:

    @staticmethod
    def is_terminal(history : str) -> bool:
        return history in ['BB', 'BC', 'CC', 'CBB', 'CBC']
        # BB = highest card wins
        # BC = p2 folds to bet, p1 gets 1
        # CC = higest card wins
        # CBB = higest card wins
        # CBC = p1 folds to p2 bet, p2 gets 1 

    @staticmethod
    def get_payoff(history : str, cards : list) -> int:
        #get payoff with respect to current player
        if history in ['BC', 'CBC']:
            # other player folded to your bet
            return 1
        else:
            id_card    : Card = cards[ len(history)      % 2]
            other_card : Card = cards[(len(history) + 1) % 2]
            payoff = 2 if 'B' in history else 1

            # from here, highest card wins 
            if id_card.value > other_card.value:
                return payoff
            else:
                return -payoff

class KuhnPokerTrainer:

    __slots__ = ("infoset_map")
    
    def __init__(self):
        self.infoset_map : Dict[str, InformationSet] = {} 
        # card_plus_history : str
        # strategy          : InformationSet

    def get_information_set(self, card_plus_history : str) -> np.ndarray:
        if card_plus_history not in self.infoset_map:
            self.infoset_map[card_plus_history] = InformationSet()
        return self.infoset_map[card_plus_history]

    def cfr(self, cards   : np.ndarray, 
                  history : str, 
                  transition_probs: np.ndarray, 
                  player: 0 or 1) -> float:

        if KuhnPoker.is_terminal(history):
            return KuhnPoker.get_payoff(history, cards)

        my_card = cards[player]
        info_set = self.get_information_set(my_card.name+history)
        counterfactual_values = np.zeros(NUM_ACTIONS) # TODO try to avoid instantiating vector every iteration

        other_player = (player + 1) % 2 # maps current player to val in {0, 1}

        # Step 1: get node value by first building up cfr values vector
        strategy = info_set.get_strategy(transition_probs[player])
        for i, action in enumerate(Actions):
            action_probability = strategy[i]

            # update transition probability to this action
            new_transition_probs = transition_probs.copy()
            new_transition_probs[player] *= action_probability

            counterfactual_values[i] = -self.cfr(cards, history+action, new_transition_probs, other_player)

        # Step 2: dot product cfr values with strategy to get expected value of node
        node_value = counterfactual_values.dot(strategy)

        # Step 3: update regret matrix for next iteration of approximation
        for i, action in enumerate(Actions):
            info_set.regret_sum[i] += transition_probs[other_player] * (counterfactual_values[i] - node_value)

        return node_value

    def train(self, iterations : int) -> float:
        root_value = 0
        for _ in range(iterations):
            cards = np.random.choice(list(Card), size=2, replace=False)
            history = ''
            transition_probs = np.ones(len(Actions))
            root_value += self.cfr(cards, history, transition_probs, 0)

        return root_value

def print_tree(card : Card, history : str, indent : int) -> None:
    if KuhnPoker.is_terminal(history):
        return
    player : str = '+' if indent%2==0 else '-'
    strategy : np.ndarray = cfr_trainer.infoset_map[card.name+history].get_average_strategy()
    print(player, ' '*indent, card.name, history, strategy)
    for action in Actions:
        print_tree(card, history+action, indent+1)
        
if __name__ == '__main__':

    # Parse args
    if len(sys.argv) < 2:
        iterations = 100000
    else:
        iterations = int(sys.argv[1])
    np.set_printoptions(precision=2, floatmode='fixed', suppress=True)

    
    cfr_trainer = KuhnPokerTrainer()
    print(f"\nRunning Kuhn Poker chance sampling CFR for {iterations} iterations")
    root_value = cfr_trainer.train(iterations)

    print(f"\nExpected average game value (for player 1): {(-1./18):.3f}")
    print(f"Computed average game value               : {(root_value / iterations):.3f}\n")

    print("We expect the bet frequency for a Jack to be between 0 and 1/3")
    print("The bet frequency of a King should be three times the one for a Jack\n")


    history=''
    for card in list(Card):
        print_tree(card, history, indent=0)
        print()

