#Counterfactual Regret Minimization implementation
#Rock Paper Scissors example
import itertools
import numpy as np

NUM_ACTIONS = 3
NUM_PLAYERS = 2
ROCK, PAPER, SCISSORS = range(NUM_ACTIONS)

A = np.array(list(itertools.product(range(NUM_ACTIONS), repeat=NUM_PLAYERS))) #all combinations of simul actions

class Player:
    def __init__(self, regrets_sum=np.zeros(NUM_ACTIONS)):
        self._strategy = np.zeros(NUM_ACTIONS)
        self.strategy_sum = np.zeros(NUM_ACTIONS)
        self.regrets_sum = regrets_sum
        self.number_of_updates = 0

    @property
    def strategy(self):
        self._strategy = np.maximum(self.regrets_sum, 0)
        self._strategy = self._strategy/sum(self._strategy) if sum(self._strategy) > 0 \
                    else np.ones(NUM_ACTIONS)/NUM_ACTIONS
        self.strategy_sum += self._strategy
        return self._strategy

    def payoff(self, my_action, other_action):
        mod_3 = (my_action - other_action) % 3
        if mod_3 == 2:
            return -1
        else:
            return mod_3

    def update_regrets(self, other_action, my_payoff):
        self.regrets_sum += np.array([self.payoff(a, other_action) - my_payoff for a in range(NUM_ACTIONS)])

    def update_strategy(self, my_action, other_action):
        my_payoff = self.payoff(my_action, other_action)
        self.update_regrets(other_action, my_payoff)
        self.number_of_updates += 1
    
def get_action(strategy):
    return np.random.choice(range(NUM_ACTIONS), p=strategy)

def expected_utility(s1, s2):
    utilities = np.array([Player.payoff(me, a1, a2) for a1, a2 in A])
    probs = np.array([s1[a1]*s2[a2] for a1, a2 in A])
    return sum(utilities*probs)

def train(iterations, me, opp):
    for _ in range(iterations):
        #⟨Get regret-matched mixed-strategy actions⟩
        my_action = get_action(me.strategy)
        opp_action = get_action(opp.strategy)
        me.update_strategy(my_action, opp_action)
        opp.update_strategy(opp_action, my_action)

    my_optimal = me.strategy_sum / me.number_of_updates
    opp_optimal = opp.strategy_sum / opp.number_of_updates
    return my_optimal, opp_optimal

if __name__ == '__main__':
    me = Player()
    opp = Player()
    #me_optimal, opp_optimal = train(1000000, me, opp)
    #print(f"my optimal strategy: {me_optimal}")
    #print(f"opponent optimal strategy: {opp_optimal}")
    mixed = np.array([1/3, 1/3, 1/3])
    pure = np.array([1, 0, 0])
    heavy = np.array([.4, .3, .3])
    print(expected_utility(mixed, mixed))
    print(expected_utility(pure, mixed))
    print(expected_utility(mixed, pure))
    print(expected_utility(heavy, pure))

