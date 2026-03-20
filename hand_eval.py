"""
Texas Hold'em Hand Evaluator

Full 52-card deck with 5-card and 7-card hand evaluation.
Returns comparable rank tuples: hand_rank(a) > hand_rank(b) means a beats b.

Hand categories (highest to lowest):
  8 = Straight Flush (Royal Flush is just the ace-high case)
  7 = Four of a Kind
  6 = Full House
  5 = Flush
  4 = Straight
  3 = Three of a Kind
  2 = Two Pair
  1 = One Pair
  0 = High Card
"""

from itertools import combinations
from collections import Counter
import random

# Card = (rank, suit)
# rank: 2..14 (2=2, ..., 10=T, 11=J, 12=Q, 13=K, 14=A)
# suit: 0=clubs, 1=diamonds, 2=hearts, 3=spades

RANK_NAMES = {2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8',
              9: '9', 10: 'T', 11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
SUIT_SYMBOLS = {0: '♣', 1: '♦', 2: '♥', 3: '♠'}

HAND_NAMES = {
    0: 'High Card', 1: 'One Pair', 2: 'Two Pair', 3: 'Three of a Kind',
    4: 'Straight', 5: 'Flush', 6: 'Full House', 7: 'Four of a Kind',
    8: 'Straight Flush'
}


def card_str(card):
    """Pretty-print a single card."""
    r, s = card
    return f"{RANK_NAMES[r]}{SUIT_SYMBOLS[s]}"


def hand_str(cards):
    """Pretty-print a list of cards."""
    return ' '.join(card_str(c) for c in cards)


def make_deck():
    """Full 52-card deck."""
    return [(r, s) for s in range(4) for r in range(2, 15)]


def eval_5card(cards):
    """
    Evaluate a 5-card poker hand.

    Returns a tuple (category, *tiebreakers) that is directly comparable.
    Higher tuple = stronger hand.
    """
    assert len(cards) == 5

    ranks = sorted([c[0] for c in cards], reverse=True)
    suits = [c[1] for c in cards]

    is_flush = len(set(suits)) == 1

    # Check for straight
    unique_ranks = sorted(set(ranks), reverse=True)
    is_straight = False
    straight_high = 0

    if len(unique_ranks) == 5:
        if unique_ranks[0] - unique_ranks[4] == 4:
            is_straight = True
            straight_high = unique_ranks[0]
        # Ace-low straight (wheel): A-2-3-4-5
        elif unique_ranks == [14, 5, 4, 3, 2]:
            is_straight = True
            straight_high = 5

    # Straight flush
    if is_straight and is_flush:
        return (8, straight_high)

    # Count rank frequencies, sort by (count desc, rank desc)
    freq = Counter(ranks)
    groups = sorted(freq.items(), key=lambda x: (x[1], x[0]), reverse=True)
    counts = [g[1] for g in groups]
    grouped_ranks = [g[0] for g in groups]

    # Four of a kind
    if counts == [4, 1]:
        return (7, grouped_ranks[0], grouped_ranks[1])

    # Full house
    if counts == [3, 2]:
        return (6, grouped_ranks[0], grouped_ranks[1])

    # Flush
    if is_flush:
        return (5, *ranks)

    # Straight
    if is_straight:
        return (4, straight_high)

    # Three of a kind
    if counts == [3, 1, 1]:
        return (3, grouped_ranks[0], grouped_ranks[1], grouped_ranks[2])

    # Two pair
    if counts == [2, 2, 1]:
        pair_ranks = sorted([g[0] for g in groups if g[1] == 2], reverse=True)
        kicker = [g[0] for g in groups if g[1] == 1][0]
        return (2, pair_ranks[0], pair_ranks[1], kicker)

    # One pair
    if counts == [2, 1, 1, 1]:
        pair_rank = grouped_ranks[0]
        kickers = sorted([g[0] for g in groups if g[1] == 1], reverse=True)
        return (1, pair_rank, *kickers)

    # High card
    return (0, *ranks)


def eval_7card(cards):
    """Best 5-card hand from 7 cards. Checks all C(7,5) = 21 combinations."""
    assert len(cards) == 7
    return max(eval_5card(list(combo)) for combo in combinations(cards, 5))


def best_hand(hole_cards, board):
    """
    Evaluate the best hand from hole cards + board.
    Works for any board size (3 = flop, 4 = turn, 5 = river).
    """
    all_cards = list(hole_cards) + list(board)
    n = len(all_cards)
    if n < 5:
        return (-1,)  # can't evaluate yet
    if n == 5:
        return eval_5card(all_cards)
    # 6 or 7 cards: check all 5-card combos
    return max(eval_5card(list(c)) for c in combinations(all_cards, 5))


def hand_name(rank_tuple):
    """Human-readable name for a hand rank tuple."""
    if not rank_tuple or rank_tuple[0] < 0:
        return "N/A"
    category = rank_tuple[0]
    name = HAND_NAMES.get(category, "Unknown")
    if category == 8 and rank_tuple[1] == 14:
        name = "Royal Flush"
    return name


def preflop_strength(hole_cards):
    """
    Compute a simple preflop hand strength score (0-100).
    Used for card bucketing. Higher = stronger.

    Based on the standard hand ranking heuristic:
    - Pairs get a big bonus
    - High cards matter
    - Suitedness adds value
    - Connectedness adds value
    """
    r1, r2 = hole_cards[0][0], hole_cards[1][0]
    s1, s2 = hole_cards[0][1], hole_cards[1][1]
    high = max(r1, r2)
    low = min(r1, r2)
    suited = (s1 == s2)
    gap = high - low

    if r1 == r2:
        # Pairs: 22=50, 33=54, ..., AA=98
        score = 50 + (r1 - 2) * 4
    else:
        # Non-pairs
        score = high * 2.5 + low * 1.0
        if suited:
            score += 4
        # Connectedness bonus
        if gap == 1:
            score += 3
        elif gap == 2:
            score += 1
        # Penalty for big gaps
        if gap >= 5:
            score -= (gap - 4) * 1

    return max(0, min(100, score))


def postflop_strength(hole_cards, board):
    """
    Compute postflop hand strength category (0-7) for bucketing.
    Uses the actual made hand rank + context about whether hole cards contribute.
    """
    if len(board) < 3:
        return 0

    rank = best_hand(hole_cards, board)
    category = rank[0]

    if category == 0:
        # High card: split by kicker quality
        if rank[1] >= 12:  # A or K high
            return 1
        return 0
    elif category == 1:
        # One pair: check if hole card makes the pair
        pair_rank = rank[1]
        hole_ranks = {hole_cards[0][0], hole_cards[1][0]}
        board_ranks = [c[0] for c in board]
        if pair_rank in hole_ranks:
            board_max = max(board_ranks)
            if pair_rank >= board_max:
                return 4  # top pair with hole card
            return 3  # underpair / middle pair with hole card
        return 2  # board pair, weak holding
    elif category == 2:
        return 5  # two pair
    elif category == 3:
        return 6  # three of a kind
    else:
        return 7  # straight, flush, full house, quads, straight flush


def has_flush_draw(hole_cards, board):
    """True if 4+ cards of one suit (but not a made flush)."""
    all_cards = list(hole_cards) + list(board)
    if len(all_cards) < 4:
        return False
    suit_counts = Counter(c[1] for c in all_cards)
    max_suit = max(suit_counts.values())
    return max_suit == 4


def has_straight_draw(hole_cards, board):
    """True if 4 cards within a span of 5 (open-ended or gutshot)."""
    all_cards = list(hole_cards) + list(board)
    ranks = sorted(set(c[0] for c in all_cards))
    # Add ace-low
    if 14 in ranks:
        ranks = [1] + ranks
    for i in range(len(ranks)):
        for j in range(i + 1, len(ranks)):
            if ranks[j] - ranks[i] <= 4:
                count = sum(1 for r in ranks[i:j+1] if ranks[i] <= r <= ranks[i] + 4)
                if count >= 4 and count < 5:  # 4 to a straight, not made
                    return True
    return False


def monte_carlo_equity(hole_cards, board, n_samples=150):
    """
    Estimate hand equity via Monte Carlo simulation.
    Deals random opponent hands and remaining board cards.
    Returns win probability (0-1).
    """
    dead = set(hole_cards) | set(board)
    remaining_deck = [c for c in make_deck() if c not in dead]

    wins = 0
    ties = 0
    total = 0
    board_cards_needed = 5 - len(board)

    for _ in range(n_samples):
        random.shuffle(remaining_deck)
        opp_hand = (remaining_deck[0], remaining_deck[1])
        sim_board = list(board) + remaining_deck[2:2 + board_cards_needed]

        my_rank = best_hand(hole_cards, sim_board)
        opp_rank = best_hand(opp_hand, sim_board)

        if my_rank > opp_rank:
            wins += 1
        elif my_rank == opp_rank:
            ties += 1
        total += 1

    return (wins + ties * 0.5) / total if total > 0 else 0.5
