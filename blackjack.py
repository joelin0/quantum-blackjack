"""
This file deals with calculating player strategies and payoffs, as well as game parameters
when the shoe contains finitely many cards.
"""

import numpy as np
from strategies import get_biased_hyperbit_sdp_discrete, get_classical_discrete, get_unlimited_comm_direct, get_payout
from itertools import combinations_with_replacement
from collections import Counter
from math import factorial

# Dictionary represents probability distribution of cards for an infinitely many 52-card decks shoe
PROBABILITIES = {i: 1./13. for i in range(1, 10)}
PROBABILITIES[10] = 4./13.
PROBABILITIES[11] = 1./13.  # used to represent soft ace


def dealer_probs():
    """
    For every possible dealer face up card, calculate the probability distribution of the dealer's final hand value.

    The dealer stands on hard 17 and soft 18.

    Resource to check: https://www.blackjackinfo.com/dealer-outcome-probabilities/#TIDH17
    NOTE: the link above distinguishes between a blackjack and a 21 after hitting. To convert,
    multiply each of the numerical values by (1 - blackjack percentage).

    THIS HAS ALREADY BEEN TESTED, SEEMS CORRECT!

    :return: A dictionary, mapping every possible (value_initial, hard) hand value pair to another dictionary.
             Each sub-dictionary maps an integer 17 <= value_final <= 21 to the probability that the dealer
             ends up with value_final, given the starting hand value (value_initial, hard). If the probabilities
             sum up to less than 1, the rest of the probability represents the probability the dealer busts.
    """
    # Pdf of any current hand (value, hard) and final value; p(v_f | v_c) where v_f = final value, v_c = current value
    probabilities = {}

    # End nodes: (value, True) for value >= 17 and (value, False) for value > 17
    # Dependencies (in order of increasing requirements):
    #               Hard values, value >= 11, possiblity of bust, no possibility of going soft with an ace (value, True) depends on (value', True) for 17 > value' > value
    #               Soft values, 17 >= value >= 11 (value, False) depends on (value', False) for 17 >= value' > value, (value', True) for 17 > value' > 11
    #               Hard values, 11 > value >= 2 , no possibility of bust, possibility of going soft with an ace (value, True) depends on (value', True) for 17 > value' > value and (value', False) for 17 >= value' > 13


    # End nodes
    for value in xrange(17, 22):
        probabilities[(value, True)] = {value: 1.0}
        if value == 17: continue  # on soft 17, dealer will still hit
        probabilities[(value, False)] = {value: 1.0}

    # Hard values, 17 > value >= 11, possibility of bust, no possibility of going soft with an ace
    for value in xrange(16, 10, -1):
        probabilities[(value, True)] = {}
        current_prob = probabilities[(value, True)]
        for next_card in xrange(1, min(10, 21-value)+1):
            next_prob = probabilities[(value + next_card, True)]
            for end_val in next_prob:
                current_prob[end_val] = current_prob.get(end_val, 0) + next_prob[end_val] * PROBABILITIES[next_card]

    # Soft values, 17 >= value >= 11
    for value in xrange(17, 10, -1):
        probabilities[(value, False)] = {}
        current_prob = probabilities[(value, False)]
        for next_card in xrange(1, 11):
            next_value = value + next_card
            hard = False
            if next_value > 21:
                next_value -= 10
                hard = True
            next_prob = probabilities[(next_value, hard)]
            for end_val in next_prob:
                current_prob[end_val] = current_prob.get(end_val, 0) + next_prob[end_val] * PROBABILITIES[next_card]

    # Hard values, 11 > value >= 2, no possibility of bust, possibility of going soft with an ace
    for value in xrange(10, 1, -1):
        probabilities[(value, True)] = {}
        current_prob = probabilities[(value, True)]
        for next_card in xrange(2, 12):
            next_value = value + next_card
            hard = (next_card != 11)
            next_prob = probabilities[(next_value, hard)]
            for end_val in next_prob:
                current_prob[end_val] = current_prob.get(end_val, 0) + next_prob[end_val] * PROBABILITIES[next_card]

    return probabilities


def get_player_payoff(dealer_end_probs):
    """
    For every player hand, maps the expected win probability if the player hits or the player stands.
    Assumes that any newly dealt card comes from an INFINITE deck.

    :param dict dealer_end_probs: a dictionary mapping a possible dealer end hand value to its probability.
                                  this assumes a given dealer face-up card.
    :return: map of player hands to the expected payoff, i.e. prob_win - prob_lose (+ 0 * prob_tie)
    """
    player_payoffs = {}  # maps a key of player hand value --> (hit payoff, stand payoff, max payoff) triples

    # Node types and dependencies
    # (value, True) for value >= 11 depends only on (value', True) for value' > value
    # (value, False) for value > 12 depends on (value', False) for value' > value and on above
    # (value, True) for value < 11 depends on above


    # probability dealer gets a certain value or lower
    dealer_cumulative_prob = {}
    tot = 0
    for val in range(17, 22):
        tot += dealer_end_probs.get(val, 0)
        dealer_cumulative_prob[val] = tot
    dealer_bust = 1 - tot

    # Hard values, 21 >= value > 10, no possibility of going soft with an ace
    for value in xrange(21, 10, -1):
        stand_payoff = (dealer_bust + dealer_cumulative_prob.get(value - 1, 0)) \
                       - (dealer_cumulative_prob.get(21, 0) - dealer_cumulative_prob.get(value, 0))
        hit_payoff = 0
        for next_card in xrange(1, 11):
            next_val = value + next_card
            if next_val <= 21:
                hit_payoff += player_payoffs[(value + next_card, True)][-1] * PROBABILITIES[next_card]
            else:
                hit_payoff -= PROBABILITIES[next_card]  # busted
        player_payoffs[(value, True)] = (hit_payoff, stand_payoff, max(stand_payoff, hit_payoff))

    # Soft values, 21 >= value > 11, no possibility of busting, possibility of going soft with an ace
    for value in xrange(21, 11, -1):
        stand_payoff = (dealer_bust + dealer_cumulative_prob.get(value - 1, 0)) \
                       - (dealer_cumulative_prob.get(21, 0) - dealer_cumulative_prob.get(value, 0))
        hit_payoff = 0
        for next_card in xrange(1, 11):
            next_value = value + next_card
            hard = False
            if next_value > 21:
                next_value -= 10
                hard = True
            hit_payoff += player_payoffs[(next_value, hard)][-1] * PROBABILITIES[next_card]
        player_payoffs[(value, False)] = (hit_payoff, stand_payoff, max(stand_payoff, hit_payoff))

    # Hard values, 10 >= value > 3, possibility of going soft with an ace
    for value in xrange(10, 3, -1):
        stand_payoff = (dealer_bust + dealer_cumulative_prob.get(value - 1, 0)) \
                       - (dealer_cumulative_prob.get(21, 0) - dealer_cumulative_prob.get(value, 0))
        hit_payoff = 0
        for next_card in xrange(2, 12):
            next_value = value + next_card
            hard = (next_card != 11)
            hit_payoff += player_payoffs[(next_value, hard)][-1] * PROBABILITIES[next_card]
        player_payoffs[(value, True)] = (hit_payoff, stand_payoff, max(stand_payoff, hit_payoff))

    return player_payoffs


def hit_stand_ev_diff(hand, shoe, dealer_hand, dealer_probabilities):
    """
    Calculates the ev difference between if Bob hits or stands,
    given the probability distribution for hitting.

    :param hand: A tuple (val, hard) representing Bob's hand
    :param shoe: A dictionary that maps card values to number in the deck. Assume to be the (finite)
                 cards remaining that the next hit from Bob will deal from.
    :param dealer_hand: A tuple (val, hard) representing the dealer's current hand
    :param dealer_probabilities: a mapping of the probability the dealer ends up with a certain hand value
    :return: Expected value of (p_win(hit) - p_lose(hit)) - (p_win(stand) - p_lose(stand))
    """
    dealer_end_probs = dealer_probabilities[dealer_hand]
    # maps a player's hand to his or her (hit_ev, stand_ev, max_ev)
    player_payoffs = get_player_payoff(dealer_end_probs)

    ev = 0  # contains weighted ev
    total = 0  # contains total weights, to normalized at the end
    val, hard = hand
    for card in shoe:
        weight = shoe[card]  # number of a card in the shoe
        total += weight
        if hard and 11 <= val <= 21:
            new_hand = (val + card, hard)
            if new_hand[0] > 21:
                ev -= weight # default loss
            else:
                ev += weight * player_payoffs[new_hand][-1]
        elif not hard and 12 <= val <= 21:
            new_val = val + card
            new_hard = False
            if new_val > 21:  # go back to hard value, take A = 1
                new_val -= 10
                new_hard = True
            ev += weight * player_payoffs[(new_val, new_hard)][-1]
        elif hard and 4 <= val <= 10:
            new_val = val + card
            new_hard = True
            if card == 1:  # go to soft value, take A = 11
                new_val += 10
                new_hard = False
            ev += weight * player_payoffs[(new_val, new_hard)][-1]
        else:
            raise RuntimeError("Should not get here: " + str(hand))
    return (1.0 * ev / total) - player_payoffs[hand][1]  # hit ev - stand ev


def get_C_matrix(ua, ub, ud, shoe=None):
    """
    The coefficient matrix of the blackjack game, given the face-up cards.

    Assumes a single deck initial shoe.

    :param int ua: Alice's face up card
    :param int ub: Bob's face up card
    :param int ud: Dealer's face up card
    :param dict shoe: The cards in the shoe, which will be used for Bob's first hit card as well as face down cards.
                      NOTE: the face up cards have NOT yet been removed from the shoe.
    :return: a numpy matrix, which gives the coefficient matrix of the game
             for a single deck shoe of blackjack, given the face up cards dealt to each actor.
    """
    if shoe is None:
        # default to a single deck shoe
        shoe = {i: 4 for i in range(1, 10)}
        shoe[10] = 16
    #
    shoe[ua] -= 1
    shoe[ub] -= 1
    shoe[ud] -= 1

    C = np.ones([10, 10])
    dealer_probabilities = dealer_probs()

    total_weight = 0
    for da in range(1, 11):
        da_weight = shoe.get(da, 0)
        C[da - 1] *= da_weight  # weighted prob of getting da
        if da_weight == 0:
            continue
        shoe[da] -= 1  # modify shoe
        for db in range(1, 11):
            db_weight = shoe.get(db, 0)
            total_weight += da_weight * db_weight
            C[da - 1][db - 1] *= db_weight  # weighted prob of getting db
            if db_weight == 0:
                continue
            shoe[db] -= 1  # modify shoe

            bob_hand = (ub + db + 10, False) if (ub == 1 or db == 1) else (ub + db, True)
            dealer_hand = (ud + 10, False) if ud == 1 else (ud, True)
            C[da - 1][db - 1] *= hit_stand_ev_diff(bob_hand, shoe, dealer_hand, dealer_probabilities)
            shoe[db] += 1  # restore shoe
        shoe[da] += 1
    return C / total_weight


def possible_advantage(C):
    first_row_scale = C * C[0,:]
    inhomogeneous_cols = first_row_scale[:,~np.all(first_row_scale >= 0, axis = 0)]
    if inhomogeneous_cols.shape[1] <= 1:
        return False

    first_col_scale = inhomogeneous_cols * inhomogeneous_cols[:,0].reshape(len(inhomogeneous_cols), 1)
    col_const = (first_col_scale >= 0)
    return inhomogeneous_cols.shape[1] > 1 and not np.all(col_const)


def generate_advantageous_configs(n_not_dealt, threshold):
    """
    Generates, in the form of ua, ub, ud, face_down_cards, configurations with a given advantage

    :param n_not_dealt: number of cards in face_down_cards
    :param threshold: threshold needed to declare advantage
    :return:
    """
    total = 0
    for _ in combinations_with_replacement(range(1, 11), n_not_dealt):
        total += 1
    total *= 100
    count = 0
    progress = min(10000, total / 10)
    print "total configs " + str(total)
    for not_dealt in combinations_with_replacement(range(1, 11), n_not_dealt):
        ua = 1
        for ub in range(1, 11):
            for ud in range(1, 11):
                shoe = {}
                for card in not_dealt:
                    shoe[card] = shoe.get(card, 0) + 1
                shoe[ua] = shoe.get(ua, 0) + 1
                shoe[ub] = shoe.get(ub, 0) + 1
                shoe[ud] = shoe.get(ud, 0) + 1
                C = get_C_matrix(ua, ub, ud, shoe)
                if possible_advantage(C):
                    nonzero_rows = C[np.any(C, axis=1)]
                    nonzero = nonzero_rows[:,~np.all(nonzero_rows == 0, axis=0)]

                    biased_S, biased_alpha, biased_D, biased_X, biased_Y = get_biased_hyperbit_sdp_discrete(
                        nonzero)
                    biased_obj = get_payout(nonzero, biased_S)

                    classical_S, classical_p, classical_alpha, classical_beta = get_classical_discrete(
                        nonzero)

                    classical_obj = get_payout(nonzero, classical_S)

                    if biased_obj - classical_obj > threshold:
                        yield ua, ub, ud, not_dealt, biased_obj, classical_obj, biased_obj - classical_obj
                count += 1
                if count % progress == 0:
                    print "finished " + str(count) + " out of " + str(total) + " configurations "


def npermutations(l):
    num = factorial(len(l))
    mults = Counter(l).values()
    den = reduce(lambda x, y: x * y, (factorial(v) for v in mults), 1)
    return num / den