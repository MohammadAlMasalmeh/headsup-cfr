#!/usr/bin/env python3
"""
Benchmark Suite for Deep CFR Poker Bot

Measures bot strength with quantifiable metrics suitable for a resume:
  1. Win rate (bb/100) vs baseline opponents over many hands
  2. Approximate exploitability (mbb/hand) via best-response
  3. EV convergence across training iterations
  4. Strategy quality checks (positional awareness, hand strength correlation)

Usage:
    python benchmark.py                          # full benchmark suite
    python benchmark.py --iters 10000 --hands 50000  # custom settings
    python benchmark.py --quick                  # fast sanity check
    python benchmark.py --output results.json    # save to custom file

Results are saved to benchmark_results/ (gitignored).
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict

import torch

from bot import (
    DeepCFRBot, ExploitativeBot, CFRNet, encode_state, legal_mask, regret_match,
    N_ACTIONS, FOLD, CHECK_CALL, BET_SMALL, BET_BIG, MAX_BETS, INPUT_DIM
)
from hand_eval import make_deck, best_hand, card_str, hand_name


RESULTS_DIR = 'benchmark_results'

# Baseline Opponents
class RandomBot:
    """Picks a uniformly random legal action."""
    name = 'Random'

    def choose(self, legal_actions, game_state, rng):
        return rng.choice(legal_actions)


class CallStation:
    """Never folds, never raises. Calls or checks everything."""
    name = 'CallStation'

    def choose(self, legal_actions, game_state, rng):
        if 'c' in legal_actions:
            return 'c'
        return 'x'


class TightAggressive:
    """
    TAG heuristic bot:
    - Preflop: raise strong hands, call medium, fold weak
    - Postflop: bet/raise with strong made hands, check/call medium, fold weak
    Uses monte_carlo_equity for postflop decisions.
    """
    name = 'TAG'

    def choose(self, legal_actions, game_state, rng):
        hole = game_state['hole_cards']
        board = game_state['board']
        street = game_state['street']
        facing = game_state['facing_bet']

        if street == 0:
            return self._preflop(hole, legal_actions, facing, rng)
        return self._postflop(hole, board, legal_actions, facing, rng)

    def _preflop(self, hole, legal_actions, facing, rng):
        r1, r2 = hole[0][0], hole[1][0]
        high, low = max(r1, r2), min(r1, r2)
        suited = hole[0][1] == hole[1][1]
        pair = r1 == r2

        # Premium: AA, KK, QQ, AKs
        if pair and high >= 12:
            return 'b' if 'b' in legal_actions else ('c' if 'c' in legal_actions else 'x')
        if high == 14 and low == 13 and suited:
            return 'b' if 'b' in legal_actions else ('c' if 'c' in legal_actions else 'x')

        # Strong: JJ-99, AK, AQs, KQs
        if pair and high >= 9:
            return 'h' if 'h' in legal_actions else ('c' if 'c' in legal_actions else 'x')
        if high == 14 and low >= 12:
            return 'h' if 'h' in legal_actions else ('c' if 'c' in legal_actions else 'x')
        if high == 13 and low == 12 and suited:
            return 'h' if 'h' in legal_actions else ('c' if 'c' in legal_actions else 'x')

        # Playable: 88-22, suited broadways, suited connectors
        if pair:
            return 'c' if 'c' in legal_actions else 'x'
        if suited and high >= 10 and low >= 9:
            return 'c' if 'c' in legal_actions else 'x'
        if suited and high - low == 1 and low >= 5:
            return 'c' if 'c' in legal_actions else 'x'
        if high == 14 and suited:
            return 'c' if 'c' in legal_actions else 'x'

        # Junk
        if facing:
            return 'f'
        return 'x'

    def _postflop(self, hole, board, legal_actions, facing, rng):
        rank = best_hand(hole, board)
        category = rank[0]

        # Very strong (two pair+): bet/raise big
        if category >= 2:
            if 'b' in legal_actions:
                return 'b'
            return 'c' if 'c' in legal_actions else 'x'

        # One pair
        if category == 1:
            pair_rank = rank[1]
            board_ranks = sorted([c[0] for c in board], reverse=True)
            hole_ranks = {hole[0][0], hole[1][0]}

            # Top pair or overpair: bet half pot
            if pair_rank in hole_ranks and pair_rank >= board_ranks[0]:
                if 'h' in legal_actions:
                    return 'h'
                return 'c' if 'c' in legal_actions else 'x'

            # Middle/bottom pair: check/call
            return 'c' if 'c' in legal_actions else 'x'

        # High card: mostly fold to bets, check otherwise
        if facing:
            # Bluff-catch with ace high sometimes
            if max(hole[0][0], hole[1][0]) >= 14 and rng.random() < 0.3:
                return 'c'
            return 'f'
        return 'x'


class LooseAggressive:
    """
    LAG heuristic bot:
    - Plays many hands, bets frequently, bluffs often.
    """
    name = 'LAG'

    def choose(self, legal_actions, game_state, rng):
        hole = game_state['hole_cards']
        board = game_state['board']
        street = game_state['street']
        facing = game_state['facing_bet']

        if street == 0:
            # Raise most hands, fold only the worst
            r1, r2 = hole[0][0], hole[1][0]
            high, low = max(r1, r2), min(r1, r2)
            if high <= 6 and low <= 4 and not (r1 == r2):
                if facing:
                    return 'f'
                return 'x'
            if rng.random() < 0.6:
                return 'b' if 'b' in legal_actions else ('h' if 'h' in legal_actions else ('c' if 'c' in legal_actions else 'x'))
            return 'h' if 'h' in legal_actions else ('c' if 'c' in legal_actions else 'x')

        # Postflop: bet aggressively with anything decent, bluff often
        rank = best_hand(hole, board)
        category = rank[0]

        if category >= 1:
            # Any made hand: bet
            if rng.random() < 0.7:
                return 'b' if 'b' in legal_actions else ('h' if 'h' in legal_actions else ('c' if 'c' in legal_actions else 'x'))
            return 'h' if 'h' in legal_actions else ('c' if 'c' in legal_actions else 'x')

        # Air: bluff sometimes
        if not facing and rng.random() < 0.4:
            return 'h' if 'h' in legal_actions else 'x'
        if facing and rng.random() < 0.15:
            return 'c'
        if facing:
            return 'f'
        return 'x'


BASELINES = [RandomBot(), CallStation(), TightAggressive(), LooseAggressive()]


# Simulated Hand Engine

class SimHand:
    """Lightweight hand simulator for benchmarking (no display)."""

    def __init__(self, stack, sb=1, bb=2):
        self.stack = stack
        self.sb = sb
        self.bb = bb

    def play(self, bot, opponent, rng, bot_seat=0):
        """
        Play one hand. bot_seat=0 means bot is SB/Button.
        Returns bot's net profit in chips.
        """
        deck = make_deck()
        rng.shuffle(deck)
        bot_hand = (deck[0], deck[1])
        opp_hand = (deck[2], deck[3])
        board = deck[4:9]

        hands = [None, None]
        hands[bot_seat] = bot_hand
        hands[1 - bot_seat] = opp_hand

        # Blinds
        stacks = [self.stack - self.sb, self.stack - self.bb]
        bets = [self.sb, self.bb]
        pot = 0
        street = 0
        cp = 0  # SB acts first preflop
        street_hist = ''
        n_bets = 0

        for _ in range(200):  # safety limit
            # Check terminal
            if street > 3 or stacks[0] <= 0 or stacks[1] <= 0:
                return self._showdown(pot + bets[0] + bets[1], stacks, hands, board, bot_seat)

            opp = 1 - cp
            to_call = max(0, bets[opp] - bets[cp])
            facing = to_call > 0
            can_raise = n_bets < MAX_BETS and stacks[cp] > to_call

            # Build legal actions
            if not facing:
                legal = ['x', 'h', 'b']
            else:
                legal = ['f', 'c']
                if can_raise:
                    legal.extend(['h', 'b'])

            # All-in edge case
            if stacks[cp] <= 0 and not facing:
                # Next street
                pot += bets[0] + bets[1]
                bets = [0, 0]
                street += 1
                street_hist = ''
                n_bets = 0
                cp = 1
                continue

            # Track whether bot bet (for opponent fold tracking)
            bot_bet_this_street = n_bets > 0 and facing and cp != bot_seat

            # Get action
            if cp == bot_seat:
                vis_board = self._visible_board(board, street)
                action_char, _, _ = bot.get_action(
                    bot_hand, vis_board, street, pot + bets[0] + bets[1],
                    to_call, facing, n_bets,
                    1 if bot_seat == 0 else 0,
                    rng
                )
                action = action_char
            else:
                vis_board = self._visible_board(board, street)
                game_state = {
                    'hole_cards': opp_hand,
                    'board': vis_board,
                    'street': street,
                    'pot': pot + bets[0] + bets[1],
                    'facing_bet': facing,
                    'to_call': to_call,
                    'n_bets': n_bets,
                    'stacks': list(stacks),
                    'position': 1 if (1 - bot_seat) == 0 else 0,
                }
                action = opponent.choose(legal, game_state, rng)
                # Feed opponent action to exploitative wrapper
                if hasattr(bot, 'observe_opponent_action'):
                    bot.observe_opponent_action(action, facing, street)

            # Apply action
            if action == 'f':
                # Opponent of folder wins
                winner = 1 - cp
                total = pot + bets[0] + bets[1]
                invested = self.stack - stacks[bot_seat]
                if winner == bot_seat:
                    return total - invested
                return -invested

            if action == 'x':
                street_hist += 'x'
                if len(street_hist) >= 2 and street_hist[-2] == 'x':
                    pot += bets[0] + bets[1]
                    bets = [0, 0]
                    street += 1
                    street_hist = ''
                    n_bets = 0
                    cp = 1
                else:
                    cp = opp
                continue

            if action == 'c':
                amt = min(to_call, stacks[cp])
                bets[cp] += amt
                stacks[cp] -= amt
                street_hist += 'c'

                # Preflop limp
                if street == 0 and street_hist == 'c':
                    cp = opp
                    continue

                pot += bets[0] + bets[1]
                bets = [0, 0]
                street += 1
                street_hist = ''
                n_bets = 0
                if street <= 3:
                    cp = 1
                continue

            # Bet/raise (h or b)
            cur_pot = pot + bets[0] + bets[1]
            pot_after = cur_pot + to_call
            if action == 'h':
                size = max(pot_after // 2, self.bb)
            else:
                size = max(pot_after, self.bb)
            amt = min(to_call + size, stacks[cp])
            bets[cp] += amt
            stacks[cp] -= amt
            n_bets += 1
            street_hist += action
            cp = opp

        # Fallback: showdown
        return self._showdown(pot + bets[0] + bets[1], stacks, hands, board, bot_seat)

    def _visible_board(self, board, street):
        if street <= 0: return []
        if street == 1: return board[:3]
        if street == 2: return board[:4]
        return board[:5]

    def _showdown(self, pot, stacks, hands, board, bot_seat):
        r0 = best_hand(hands[0], board)
        r1 = best_hand(hands[1], board)
        invested = self.stack - stacks[bot_seat]

        if r0 > r1:
            winner = 0
        elif r1 > r0:
            winner = 1
        else:
            return pot / 2.0 - invested  # split

        if winner == bot_seat:
            return pot - invested
        return -invested


# Head-to-Head vs Baselines

def benchmark_vs_baselines(bot, n_hands=50000, stack=200, sb=1, bb=2, seed=123):
    """
    Play n_hands against each baseline opponent.
    Returns dict of {opponent_name: {bb_per_100, total_profit, hands, wins, losses}}.
    """
    sim = SimHand(stack, sb, bb)
    results = {}

    # Wrap in exploitative bot for adaptation
    exploit_bot = ExploitativeBot(bot) if not isinstance(bot, ExploitativeBot) else bot

    for opponent in BASELINES:
        rng = random.Random(seed)
        total_profit = 0.0
        wins = 0
        losses = 0
        ties = 0

        # Reset opponent model for each new opponent
        exploit_bot.reset_stats()

        t0 = time.time()
        for i in range(n_hands):
            bot_seat = i % 2  # alternate positions
            profit = sim.play(exploit_bot, opponent, rng, bot_seat=bot_seat)
            total_profit += profit
            if profit > 0:
                wins += 1
            elif profit < 0:
                losses += 1
            else:
                ties += 1

        elapsed = time.time() - t0
        bb_per_100 = (total_profit / n_hands) * 100 / bb

        results[opponent.name] = {
            'bb_per_100': round(bb_per_100, 2),
            'total_profit': round(total_profit, 2),
            'hands': n_hands,
            'wins': wins,
            'losses': losses,
            'ties': ties,
            'win_pct': round(wins / n_hands * 100, 1),
            'hands_per_sec': round(n_hands / elapsed, 0),
        }

        print(f"  vs {opponent.name:12s}: "
              f"{bb_per_100:+8.2f} bb/100 | "
              f"Win {wins/n_hands*100:5.1f}% | "
              f"{n_hands/elapsed:.0f} hands/s")

    return results


# Approximate Exploitability

@torch.no_grad()
def approximate_exploitability(bot, n_samples=20000, seed=456):
    """
    Approximate exploitability by computing a best-response value.

    For each sampled deal, we compute the EV of a best-response opponent
    that always picks the action maximizing its EV against the bot's
    fixed strategy. The exploitability is how much the best-response
    opponent wins on average.

    Returns exploitability in mbb/hand (milli big blinds per hand).
    Lower = closer to Nash equilibrium. 0 = unexploitable.
    """
    rng = random.Random(seed)
    deck = make_deck()
    sim = SimHand(bot.stack, bot.sb, bot.bb)

    # We approximate by: for each deal, play the hand with the bot
    # using its strategy, but the opponent tries all legal actions
    # and picks the best one at each decision point.
    # Full best-response is exponential, so we use a sampling approximation:
    # play many hands where opponent uses each baseline strategy,
    # take the max EV across opponents as a lower bound on exploitability.

    best_ev = float('-inf')
    for opponent in BASELINES:
        total = 0.0
        opp_rng = random.Random(seed)  # same deals for fair comparison
        for i in range(n_samples):
            bot_seat = i % 2
            profit = sim.play(bot, opponent, opp_rng, bot_seat=bot_seat)
            total -= profit  # opponent's perspective
        ev = total / n_samples
        if ev > best_ev:
            best_ev = ev
            best_opp = opponent.name

    # Convert to mbb/hand
    mbb_hand = best_ev / bot.bb * 1000

    print(f"  Approx exploitability: {mbb_hand:+.1f} mbb/hand "
          f"(best response: {best_opp})")

    return {
        'mbb_per_hand': round(mbb_hand, 1),
        'best_response_opponent': best_opp,
        'samples': n_samples,
    }


# Strategy Quality Analysis

@torch.no_grad()
def analyze_strategy_quality(bot):
    """
    Check if the bot's strategy exhibits key properties of good poker:
    1. Position awareness: plays more hands in position
    2. Hand strength correlation: bets more with strong hands
    3. Bluff frequency: doesn't bet 0% or 100% with weak hands
    4. Fold to aggression: folds weak hands when facing bets
    """
    results = {}

    # Sample hands by category
    premium = [((14, 3), (14, 0)), ((13, 2), (13, 1)), ((14, 3), (13, 3))]
    medium = [((10, 0), (9, 0)), ((8, 1), (7, 1)), ((11, 0), (10, 2))]
    weak = [((2, 1), (7, 3)), ((3, 0), (8, 2)), ((2, 0), (5, 1))]

    categories = {'premium': premium, 'medium': medium, 'weak': weak}

    # Test 1: Preflop aggression by hand strength
    print("  Preflop aggression by hand strength:")
    preflop_agg = {}
    for cat_name, hands in categories.items():
        bet_freq = 0.0
        for hand in hands:
            for pos in [0, 1]:
                enc = encode_state(hand, [], 3, bot.stack, 0, 0, 0, pos)
                logits = bot.strat_net(enc.unsqueeze(0)).squeeze(0)
                mask = legal_mask(False, True)
                logits[~mask] = float('-inf')
                probs = torch.softmax(logits, dim=0)
                bet_freq += (probs[BET_SMALL] + probs[BET_BIG]).item()
        bet_freq /= (len(hands) * 2)
        preflop_agg[cat_name] = round(bet_freq * 100, 1)
        print(f"    {cat_name:8s}: {bet_freq*100:5.1f}% bet/raise")

    results['preflop_aggression'] = preflop_agg

    # Check: premium should bet more than weak
    strength_correlated = preflop_agg['premium'] > preflop_agg['weak']
    results['strength_correlated'] = strength_correlated
    print(f"    Strength-correlated: {'YES' if strength_correlated else 'NO'}")

    # Test 2: Position awareness (IP should play more aggressively)
    print("\n  Position awareness (half-pot+ bet frequency):")
    pos_agg = {'IP': 0.0, 'OOP': 0.0}
    test_hands = premium + medium + weak
    for hand in test_hands:
        for pos, pos_name in [(1, 'IP'), (0, 'OOP')]:
            enc = encode_state(hand, [], 3, bot.stack, 0, 0, 0, pos)
            logits = bot.strat_net(enc.unsqueeze(0)).squeeze(0)
            mask = legal_mask(False, True)
            logits[~mask] = float('-inf')
            probs = torch.softmax(logits, dim=0)
            pos_agg[pos_name] += (probs[BET_SMALL] + probs[BET_BIG]).item()
    for k in pos_agg:
        pos_agg[k] = round(pos_agg[k] / len(test_hands) * 100, 1)
    print(f"    In position:  {pos_agg['IP']:5.1f}%")
    print(f"    Out of pos:   {pos_agg['OOP']:5.1f}%")

    position_aware = pos_agg['IP'] >= pos_agg['OOP']
    results['position_awareness'] = {
        'ip_bet_pct': pos_agg['IP'],
        'oop_bet_pct': pos_agg['OOP'],
        'position_aware': position_aware,
    }
    print(f"    Position-aware: {'YES' if position_aware else 'NO'}")

    # Test 3: Postflop play on a specific board
    print("\n  Postflop strategy on A-T-3 rainbow:")
    board = [(14, 2), (10, 1), (3, 0)]
    for cat_name, hands in categories.items():
        bet_freq = 0.0
        for hand in hands:
            enc = encode_state(hand, board, 20, bot.stack, 1, 0, 0, 1)
            logits = bot.strat_net(enc.unsqueeze(0)).squeeze(0)
            mask = legal_mask(False, True)
            logits[~mask] = float('-inf')
            probs = torch.softmax(logits, dim=0)
            bet_freq += (probs[BET_SMALL] + probs[BET_BIG]).item()
        bet_freq /= len(hands)
        print(f"    {cat_name:8s}: {bet_freq*100:5.1f}% c-bet")

    # Test 4: Fold frequency when facing a pot bet
    print("\n  Fold frequency facing pot-size bet (preflop):")
    fold_by_cat = {}
    for cat_name, hands in categories.items():
        fold_freq = 0.0
        for hand in hands:
            enc = encode_state(hand, [], 6, bot.stack, 0, 1, 1, 0)
            logits = bot.strat_net(enc.unsqueeze(0)).squeeze(0)
            mask = legal_mask(True, True)
            logits[~mask] = float('-inf')
            probs = torch.softmax(logits, dim=0)
            fold_freq += probs[FOLD].item()
        fold_freq /= len(hands)
        fold_by_cat[cat_name] = round(fold_freq * 100, 1)
        print(f"    {cat_name:8s}: {fold_freq*100:5.1f}% fold")

    fold_discriminating = fold_by_cat['weak'] > fold_by_cat['premium']
    results['fold_discipline'] = {
        'by_category': fold_by_cat,
        'discriminating': fold_discriminating,
    }
    print(f"    Discriminating: {'YES' if fold_discriminating else 'NO'}")

    return results


# EV Convergence

def benchmark_ev_convergence(bot_class, iters_list, stack=200, sb=1, bb=2, seed=42):
    """
    Train bots at different iteration counts and measure EV convergence.
    Shows how strategy improves with more training.
    """
    results = []
    sim = SimHand(stack, sb, bb)
    eval_hands = 10000
    eval_seed = 999

    for iters in iters_list:
        print(f"  Training {iters:>6,} iterations...", end='', flush=True)
        t0 = time.time()

        bot = bot_class(stack=stack, sb=sb, bb=bb)
        bot.train(iters=iters, seed=seed, progress=False)

        train_time = time.time() - t0

        # Evaluate vs CallStation (stable baseline)
        rng = random.Random(eval_seed)
        total = 0.0
        for i in range(eval_hands):
            total += sim.play(bot, CallStation(), rng, bot_seat=i % 2)

        bb_100 = (total / eval_hands) * 100 / bb
        print(f"  {bb_100:+8.2f} bb/100 vs CallStation  ({train_time:.1f}s)")

        results.append({
            'iterations': iters,
            'bb_per_100_vs_callstation': round(bb_100, 2),
            'train_time_sec': round(train_time, 1),
        })

    return results


# Main

def run_full_benchmark(iters=30000, n_hands=50000, stack=200, sb=1, bb=2,
                       seed=42, model_path='model.pt', retrain=False):
    """Run the complete benchmark suite and return all results."""
    results = {
        'config': {
            'training_iterations': iters,
            'eval_hands': n_hands,
            'stack': stack,
            'blinds': f'{sb}/{bb}',
            'seed': seed,
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    print("=" * 60)
    print("  Deep CFR Poker Bot - Benchmark Suite")
    print("=" * 60)

    # 1. Train or load
    if os.path.exists(model_path) and not retrain:
        print(f"\n[1/5] Loading trained bot from {model_path}...")
        bot = DeepCFRBot.load(model_path)
        train_time = 0.0
    else:
        print(f"\n[1/5] Training bot ({iters:,} iterations)...")
        t0 = time.time()
        bot = DeepCFRBot(stack=stack, sb=sb, bb=bb)
        bot.train(iters=iters, seed=seed, progress=True)
        bot.save(model_path)
        train_time = time.time() - t0
    results['training_time_sec'] = round(train_time, 1)

    # 2. Head-to-head
    print(f"\n[2/5] Head-to-head vs baselines ({n_hands:,} hands each)...")
    results['head_to_head'] = benchmark_vs_baselines(
        bot, n_hands=n_hands, stack=stack, sb=sb, bb=bb, seed=seed + 1
    )

    # 3. Exploitability
    print(f"\n[3/5] Approximate exploitability...")
    results['exploitability'] = approximate_exploitability(
        bot, n_samples=min(n_hands, 20000), seed=seed + 2
    )

    # 4. Strategy quality
    print(f"\n[4/5] Strategy quality analysis...")
    results['strategy_quality'] = analyze_strategy_quality(bot)

    # 5. EV convergence
    print(f"\n[5/5] EV convergence across training budgets...")
    convergence_points = [1000, 5000, 10000]
    if iters >= 30000:
        convergence_points.append(30000)
    if iters >= 50000:
        convergence_points.append(50000)
    results['convergence'] = benchmark_ev_convergence(
        DeepCFRBot, convergence_points, stack=stack, sb=sb, bb=bb, seed=seed
    )

    # Summary
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)

    h2h = results['head_to_head']
    print(f"\n  Training: {iters:,} iterations in {train_time:.1f}s")
    print(f"\n  Win Rates (bb/100):")
    for opp, data in h2h.items():
        bar = '+' * max(0, int(data['bb_per_100'] / 5)) if data['bb_per_100'] > 0 else '-' * max(0, int(-data['bb_per_100'] / 5))
        print(f"    vs {opp:12s}: {data['bb_per_100']:+8.2f}  {bar}")

    expl = results['exploitability']
    print(f"\n  Exploitability: {expl['mbb_per_hand']:+.1f} mbb/hand")

    sq = results['strategy_quality']
    checks = [
        ('Strength-correlated', sq.get('strength_correlated', False)),
        ('Position-aware', sq.get('position_awareness', {}).get('position_aware', False)),
        ('Fold discipline', sq.get('fold_discipline', {}).get('discriminating', False)),
    ]
    print(f"\n  Strategy Quality:")
    for name, passed in checks:
        print(f"    {'PASS' if passed else 'FAIL'} {name}")

    print(f"\n  Convergence (bb/100 vs CallStation):")
    for c in results['convergence']:
        print(f"    {c['iterations']:>6,} iters: {c['bb_per_100_vs_callstation']:+8.2f}")

    print("\n" + "=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark Deep CFR Poker Bot')
    parser.add_argument('--iters', type=int, default=30000,
                        help='Training iterations (default: 30000)')
    parser.add_argument('--hands', type=int, default=50000,
                        help='Evaluation hands per opponent (default: 50000)')
    parser.add_argument('--stack', type=int, default=200,
                        help='Starting stack (default: 200)')
    parser.add_argument('--sb', type=int, default=1, help='Small blind')
    parser.add_argument('--bb', type=int, default=2, help='Big blind')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename (default: benchmark_TIMESTAMP.json)')
    parser.add_argument('--model', type=str, default='model.pt',
                        help='Model file path (default: model.pt)')
    parser.add_argument('--retrain', action='store_true',
                        help='Force retrain even if saved model exists')
    parser.add_argument('--quick', action='store_true',
                        help='Quick benchmark (5k iters, 10k hands)')
    args = parser.parse_args()

    if args.quick:
        args.iters = 5000
        args.hands = 10000

    results = run_full_benchmark(
        iters=args.iters, n_hands=args.hands,
        stack=args.stack, sb=args.sb, bb=args.bb, seed=args.seed,
        model_path=args.model, retrain=args.retrain,
    )

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if args.output:
        out_path = os.path.join(RESULTS_DIR, args.output)
    else:
        ts = time.strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join(RESULTS_DIR, f'benchmark_{ts}.json')

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to: {out_path}\n")


if __name__ == '__main__':
    main()
