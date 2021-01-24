#Monte Carlo Based Counterfactual Regret Optimization with Kuhn Poker
#Players are each dealt 1 card from a deck of 3, actions=[check, bet] 

import sys
import numpy as np

Actions = ['B', 'C'] #bet/call check/fold

class InformationSet:
    def __init__(self):
        self.strategy_sum = np.zeros(len(Actions))
        self.regret_sum = np.zeros(len(Actions))
        self.num_actions = len(Actions)

    def normalize(self, strategy):
        if sum(strategy) > 0:
            return strategy/sum(strategy)
        else:
            return np.ones(self.num_actions)/self.num_actions

    def get_strategy(self, transition_prob):
        strategy = np.maximum(self.regret_sum, 0)
        strategy = self.normalize(strategy)

        self.strategy_sum += transition_prob * strategy
        return strategy

    def get_average_strategy(self):
        return self.normalize(self.strategy_sum)

class KuhnPoker:
    @staticmethod
    def is_terminal(history):
        return history in ['BB', 'BC', 'CC', 'CBB', 'CBC']

    @staticmethod
    def get_payoff(history, cards):
        #get payoff with respect to current player
        if history in ['BC', 'CBC']:
            return 1
        else:
            payoff = 2 if 'B' in history else 1
            active_player = len(history) % 2
            player_card = cards[active_player]
            opp_card = cards[(active_player + 1) % 2]
            if player_card == 'K' or opp_card == 'J':
                return payoff
            else:
                return -payoff

class KuhnPokerTrainer:
    def __init__(self):
        self.infoset_map = {}

    def get_information_set(self, card_plus_history):
        if card_plus_history not in self.infoset_map:
            self.infoset_map[card_plus_history] = InformationSet()
        return self.infoset_map[card_plus_history]

    def cfr(self, cards, history, transition_probs, active_player):
        if KuhnPoker.is_terminal(history):
            return KuhnPoker.get_payoff(history, cards)

        my_card = cards[active_player]
        info_set = self.get_information_set(my_card+history)

        strategy = info_set.get_strategy(transition_probs[active_player])
        opp = (active_player + 1) % 2
        counterfactual_values = np.zeros(len(Actions))

        for i, action in enumerate(Actions):
            action_probability = strategy[i]

            new_transition_probs = transition_probs.copy()
            new_transition_probs[active_player] *= action_probability

            counterfactual_values[i] = -self.cfr(cards, history+action, new_transition_probs, opp)

        node_value = counterfactual_values.dot(strategy)
        for i, action in enumerate(Actions):
            info_set.regret_sum[i] += transition_probs[opp] * (counterfactual_values[i] - node_value)

        return node_value

    def train(self, iterations):
        util = 0
        kuhn_cards = ['J', 'Q', 'K']
        for _ in range(iterations):
            cards = np.random.choice(kuhn_cards, size=2, replace=False)
            history = ''
            transition_probs = np.ones(len(Actions))
            util += self.cfr(cards, history, transition_probs, 0)

        return util

if __name__ == '__main__':
    if len(sys.argv) < 2:
        iterations = 100000
    else:
        iterations = int(sys.argv[1])
    np.set_printoptions(precision=2, floatmode='fixed', suppress=True)

    cfr_trainer = KuhnPokerTrainer()
    print("starting training")
    util = cfr_trainer.train(iterations)

    print(f"\nRunning Kuhn Poker chance sampling CFR for {iterations} iterations")
    print(f"\nExpected average game value (for player 1): {(-1./18):.3f}")
    print(f"Computed average game value               : {(util / iterations):.3f}\n")

    print("We expect the bet frequency for a Jack to be between 0 and 1/3")
    print("The bet frequency of a King should be three times the one for a Jack\n")


    history=''
    cards = ['J', 'Q', 'K']
    def print_tree(history, indent):
        if KuhnPoker.is_terminal(history[1:]):
            return
        player = '+' if indent%2==0 else '-'
        strategy = cfr_trainer.infoset_map[history].get_average_strategy()
        print(player, ' '*indent, history, strategy)
        for action in Actions:
            print_tree(history+action, indent+1)

    for card in cards:
        print_tree(card, 0)

