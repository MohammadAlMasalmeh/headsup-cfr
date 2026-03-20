"""
Deep CFR Bot for Heads-Up No-Limit Texas Hold'em

Replaces tabular MCCFR with neural function approximation (Deep CFR).
Instead of bucketing cards into 5 bins, the network learns directly
from raw card features plus computed hand strength signals.

Architecture (from the Deep CFR paper, Steinberger 2019):
  - Advantage network: predicts counterfactual regrets per action
  - Strategy network: predicts average strategy action probabilities
  - Reservoir buffer: maintains uniform sample over all training data
  - External Sampling MCCFR drives the data collection

Action space (fixed 4 outputs, masked per situation):
  0 = fold
  1 = check / call
  2 = bet/raise half pot
  3 = bet/raise pot

Requires: torch, numpy
"""

import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from hand_eval import (
    make_deck, best_hand, preflop_strength, postflop_strength,
    has_flush_draw, has_straight_draw
)

# Constants
N_ACTIONS = 4
FOLD, CHECK_CALL, BET_SMALL, BET_BIG = 0, 1, 2, 3

# 52 hole + 52 board + 8 hand features + 4 street + 4 game state = 120
INPUT_DIM = 120
MAX_BETS = 4      # max bets+raises per street

# Map network output index -> game action char
IDX_TO_CHAR = {FOLD: 'f', CHECK_CALL: 'x', BET_SMALL: 'h', BET_BIG: 'b'}
IDX_TO_CHAR_FACING = {FOLD: 'f', CHECK_CALL: 'c', BET_SMALL: 'h', BET_BIG: 'b'}


# State Encoding
def encode_cards(cards):
    """Encode a list of cards as a 52-dim binary vector."""
    v = torch.zeros(52)
    for rank, suit in cards:
        idx = (rank - 2) * 4 + suit  # rank 2-14 -> 0-12, suit 0-3
        v[idx] = 1.0
    return v


def encode_state(hole_cards, board, pot, stack, street, to_call, n_bets, position):
    """
    Encode the full game state as a 120-dim tensor.

    Layout:
      [0:52]    hole cards (binary)
      [52:104]  board cards (binary)
      [104]     preflop hand strength (0-1)
      [105]     postflop hand category (0-1, category/8)
      [106]     top pair or better indicator
      [107]     overpair indicator
      [108]     flush draw indicator
      [109]     straight draw indicator
      [110]     stack-to-pot ratio (capped at 1.0)
      [111]     pot commitment (invested / stack)
      [112:116] street (one-hot, 4 dims)
      [116]     pot / (2 * stack)
      [117]     to_call / max(pot, 1)
      [118]     n_bets / MAX_BETS
      [119]     position (0 = OOP, 1 = IP)
    """
    enc = torch.zeros(INPUT_DIM)

    # Hole cards (binary)
    for r, s in hole_cards:
        enc[(r - 2) * 4 + s] = 1.0

    # Board cards (binary)
    for r, s in board:
        enc[52 + (r - 2) * 4 + s] = 1.0

    # --- Hand strength features ---

    # Preflop strength (0-1)
    enc[104] = preflop_strength(hole_cards) / 100.0

    # Postflop features (only when board exists)
    if len(board) >= 3:
        rank = best_hand(hole_cards, board)
        category = rank[0]  # 0=high card .. 8=straight flush
        enc[105] = category / 8.0

        # Top pair or better
        hole_ranks = {hole_cards[0][0], hole_cards[1][0]}
        board_ranks = [c[0] for c in board]
        board_max = max(board_ranks)

        if category >= 2:
            # Two pair or better
            enc[106] = 1.0
        elif category == 1:
            pair_rank = rank[1]
            if pair_rank in hole_ranks and pair_rank >= board_max:
                enc[106] = 1.0  # top pair with hole card

        # Overpair (pocket pair > highest board card)
        if hole_cards[0][0] == hole_cards[1][0] and hole_cards[0][0] > board_max:
            enc[107] = 1.0

        # Draw indicators
        if category < 5:  # no made flush yet
            if has_flush_draw(hole_cards, board):
                enc[108] = 1.0
        if category < 4:  # no made straight yet
            if has_straight_draw(hole_cards, board):
                enc[109] = 1.0
    else:
        # Preflop: pair indicator as a stand-in
        if hole_cards[0][0] == hole_cards[1][0]:
            enc[106] = 1.0

    # Stack-to-pot ratio (capped at 1.0 for normalization)
    effective_stack = min(stack, stack)  # heads-up, both stacks same at start
    spr = effective_stack / max(pot, 1.0)
    enc[110] = min(spr / 20.0, 1.0)  # normalize: SPR 20 -> 1.0

    # Pot commitment (fraction of stack invested)
    invested = max(stack - effective_stack, 0)
    # During traversal, we don't track individual invested easily,
    # so use pot ratio as proxy
    enc[111] = pot / max(2.0 * stack, 1.0)

    # --- Game state features ---

    # Street one-hot
    enc[112 + min(street, 3)] = 1.0

    # Pot ratio
    enc[116] = pot / max(2.0 * stack, 1.0)

    # To-call ratio
    enc[117] = to_call / max(float(pot), 1.0)

    # Bets
    enc[118] = n_bets / MAX_BETS

    # Position
    enc[119] = float(position)

    return enc

# Legal Action Masking
def legal_mask(facing_bet, can_raise):
    """
    Returns a boolean mask over the 4 action slots.

    Not facing bet: check(1), bet_half(2), bet_pot(3)
    Facing + can raise: fold(0), call(1), raise_half(2), raise_pot(3)
    Facing + no raise: fold(0), call(1)
    """
    m = torch.zeros(N_ACTIONS, dtype=torch.bool)
    if not facing_bet:
        m[CHECK_CALL] = True
        m[BET_SMALL] = True
        m[BET_BIG] = True
    else:
        m[FOLD] = True
        m[CHECK_CALL] = True
        if can_raise:
            m[BET_SMALL] = True
            m[BET_BIG] = True
    return m


def regret_match(advantages, mask):
    """
    Regret matching on masked advantage predictions.
    Positive advantages -> strategy proportions.
    Returns probability vector over all N_ACTIONS (0 for illegal).
    """
    masked = advantages.clone()
    masked[~mask] = 0.0
    positive = torch.clamp(masked, min=0.0)
    total = positive.sum()
    if total > 0:
        return positive / total
    # Uniform over legal actions
    u = mask.float()
    return u / u.sum()

# Neural Networks
class CFRNet(nn.Module):
    """
    Network for advantage and strategy prediction.

    3 hidden layers of 128 units (~50k params).
    Sized to handle the enriched 120-dim state encoding with
    hand strength features.
    """

    def __init__(self, input_dim=INPUT_DIM, hidden=128, output_dim=N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# Reservoir Buffer
class ReservoirBuffer:
    """
    Fixed-size buffer with reservoir sampling.

    Maintains a uniform random sample over all items ever added.
    When the buffer is full, each new item replaces a random existing
    item with probability (max_size / total_seen).
    """

    def __init__(self, max_size=500_000):
        self.max_size = max_size
        self.buffer = []
        self.total_seen = 0

    def add(self, encoding, target, iteration):
        self.total_seen += 1
        item = (encoding.numpy(), target.numpy(), iteration)

        if len(self.buffer) < self.max_size:
            self.buffer.append(item)
        else:
            idx = random.randint(0, self.total_seen - 1)
            if idx < self.max_size:
                self.buffer[idx] = item

    def get_training_data(self, max_samples=None):
        import numpy as np

        n = len(self.buffer)
        if n == 0:
            return None, None, None

        if max_samples and max_samples < n:
            samples = random.sample(self.buffer, max_samples)
        else:
            samples = self.buffer

        encodings = torch.from_numpy(np.stack([s[0] for s in samples]))
        targets = torch.from_numpy(np.stack([s[1] for s in samples]))
        weights = torch.tensor([s[2] for s in samples], dtype=torch.float32)

        return encodings, targets, weights

    def __len__(self):
        return len(self.buffer)


# Deep CFR Bot
class DeepCFRBot:
    """
    Deep CFR for Heads-Up No-Limit Texas Hold'em.

    Training procedure (per the Deep CFR paper):
      1. Deal a hand (real 52-card deck)
      2. Traverse the abstract game tree with external sampling
      3. Collect (encoding, regrets) for advantage net
      4. Collect (encoding, strategy) for strategy net
      5. Periodically retrain advantage net (few epochs, unweighted)
      6. After all iterations, train strategy net (iteration-weighted)

    Play:
      - Encode game state -> strategy net forward pass -> masked softmax -> sample
    """

    def __init__(self, stack=200, sb=1, bb=2, device='cpu'):
        self.stack = stack
        self.sb = sb
        self.bb = bb
        self.device = torch.device(device)

        # Networks
        self.adv_net = CFRNet().to(self.device)
        self.strat_net = CFRNet().to(self.device)

        # Persistent optimizers (preserve momentum across retrains)
        self.adv_optimizer = torch.optim.Adam(self.adv_net.parameters(), lr=5e-4)
        self.strat_optimizer = None  # created at final training

        # Reservoir buffers
        self.adv_buffer = ReservoirBuffer(500_000)
        self.strat_buffer = ReservoirBuffer(500_000)


    def _bet_amount(self, action_idx, pot, bets, stacks, cp):
        """Compute chip amount for an abstract action during traversal."""
        opp = 1 - cp
        to_call = bets[opp] - bets[cp]
        current_pot = pot + bets[0] + bets[1]

        if action_idx == FOLD or action_idx == CHECK_CALL:
            if action_idx == CHECK_CALL and to_call > 0:
                return min(to_call, stacks[cp])
            return 0
        elif action_idx == BET_SMALL:
            pot_after = current_pot + to_call
            size = max(pot_after // 2, self.bb)
            return min(to_call + size, stacks[cp])
        elif action_idx == BET_BIG:
            pot_after = current_pot + to_call
            size = max(pot_after, self.bb)
            return min(to_call + size, stacks[cp])
        return 0

    @torch.no_grad()
    def _traverse(self, street, pot, stacks, bets, cp, st_hist, n_bets,
                  traverser, hands, board, iteration, rng):
        """
        External sampling MCCFR traversal with neural advantage network.
        Returns utility for the traversing player.
        """
        opp = 1 - cp
        to_call = bets[opp] - bets[cp]
        facing = to_call > 0
        can_raise = n_bets < MAX_BETS and stacks[cp] > to_call

        # All-in with nothing to do: advance
        if stacks[cp] <= 0 and not facing:
            return self._next_street(street, pot, stacks, bets,
                                     traverser, hands, board, iteration, rng)

        # Get legal mask
        mask = legal_mask(facing, can_raise)
        legal_indices = mask.nonzero(as_tuple=True)[0].tolist()
        n_legal = len(legal_indices)

        if n_legal == 0:
            return self._next_street(street, pot, stacks, bets,
                                     traverser, hands, board, iteration, rng)

        # Build visible board for this street
        if street == 0:
            vis_board = []
        elif street == 1:
            vis_board = board[:3]
        elif street == 2:
            vis_board = board[:4]
        else:
            vis_board = board[:5]

        # Encode state and get advantage predictions
        enc = encode_state(
            hands[cp], vis_board, pot + bets[0] + bets[1],
            self.stack, street, to_call, n_bets, 1 if cp == 0 else 0
        )
        adv = self.adv_net(enc.unsqueeze(0)).squeeze(0)
        strategy = regret_match(adv, mask)

        # Store strategy sample
        self.strat_buffer.add(enc, strategy.clone(), iteration)

        if cp == traverser:
            # Explore ALL legal actions, compute counterfactual values
            action_values = torch.zeros(N_ACTIONS)

            for a_idx in legal_indices:
                val = self._apply_action(
                    a_idx, street, pot, stacks, bets, cp, st_hist,
                    n_bets, traverser, hands, board, iteration, rng
                )
                action_values[a_idx] = val

            # Node value (expected value under current strategy)
            node_value = (strategy * action_values).sum().item()

            # Counterfactual regrets
            regrets = action_values.clone()
            for a_idx in legal_indices:
                regrets[a_idx] -= node_value
            regrets[~mask] = 0.0

            # Store advantage sample
            self.adv_buffer.add(enc, regrets, iteration)

            return node_value

        else:
            # OPPONENT: sample one action from strategy
            probs = strategy[legal_indices].tolist()
            total = sum(probs)
            if total > 0:
                probs = [p / total for p in probs]
            else:
                probs = [1.0 / n_legal] * n_legal

            r = rng.random()
            cum = 0.0
            sampled_idx = legal_indices[-1]
            for i, p in enumerate(probs):
                cum += p
                if r < cum:
                    sampled_idx = legal_indices[i]
                    break

            return self._apply_action(
                sampled_idx, street, pot, stacks, bets, cp, st_hist,
                n_bets, traverser, hands, board, iteration, rng
            )

    def _apply_action(self, action_idx, street, pot, stacks, bets, cp,
                      st_hist, n_bets, traverser, hands, board, iteration, rng):
        """Apply an abstract action and recurse."""
        opp = 1 - cp
        s = [stacks[0], stacks[1]]
        b = [bets[0], bets[1]]
        to_call = b[opp] - b[cp]

        # -- Fold --
        if action_idx == FOLD:
            total = pot + b[0] + b[1]
            inv_t = self.stack - s[traverser]
            if opp == traverser:
                return total - inv_t
            return -(self.stack - s[traverser])

        # -- Check / Call --
        if action_idx == CHECK_CALL:
            if to_call > 0:
                amt = min(to_call, s[cp])
                b[cp] += amt
                s[cp] -= amt

                # Preflop limp
                if street == 0 and st_hist == '':
                    return self._traverse(street, pot, s, b, opp, 'c',
                                          n_bets, traverser, hands, board, iteration, rng)
                return self._next_street(street, pot, s, b,
                                         traverser, hands, board, iteration, rng)
            else:
                if st_hist and st_hist[-1] == 'x':
                    return self._next_street(street, pot, s, b,
                                             traverser, hands, board, iteration, rng)
                return self._traverse(street, pot, s, b, opp, st_hist + 'x',
                                      n_bets, traverser, hands, board, iteration, rng)

        # -- Bet / Raise --
        amt = self._bet_amount(action_idx, pot, bets, stacks, cp)
        b[cp] += amt
        s[cp] -= amt

        tag = 'h' if action_idx == BET_SMALL else 'b'
        return self._traverse(street, pot, s, b, opp, st_hist + tag,
                              n_bets + 1, traverser, hands, board, iteration, rng)

    def _next_street(self, street, pot, stacks, bets,
                     traverser, hands, board, iteration, rng):
        new_pot = pot + bets[0] + bets[1]
        ns = street + 1

        if ns > 3 or stacks[0] <= 0 or stacks[1] <= 0:
            return self._showdown(new_pot, stacks, traverser, hands, board)

        # BB (seat 1) acts first postflop
        return self._traverse(ns, new_pot, stacks, [0, 0], 1, '',
                              0, traverser, hands, board, iteration, rng)

    def _showdown(self, pot, stacks, traverser, hands, board):
        """Evaluate real hands at showdown."""
        r0 = best_hand(hands[0], board)
        r1 = best_hand(hands[1], board)
        inv = [self.stack - stacks[i] for i in range(2)]

        if r0 > r1:
            p0, p1 = pot - inv[0], -inv[1]
        elif r1 > r0:
            p0, p1 = -inv[0], pot - inv[1]
        else:
            h = pot / 2.0
            p0, p1 = h - inv[0], h - inv[1]

        return p0 if traverser == 0 else p1


    def _train_network(self, net, optimizer, buffer, epochs, batch_size,
                       use_iteration_weight=False, is_strategy=False):
        """
        Train a network on the reservoir buffer.

        For advantage net: MSE loss on regret targets (unweighted, persistent optimizer)
        For strategy net: Cross-entropy loss (iteration-weighted per Deep CFR)
        """
        data = buffer.get_training_data()
        if data[0] is None or len(buffer) < batch_size:
            return 0.0

        encodings, targets, iterations = data
        encodings = encodings.to(self.device)
        targets = targets.to(self.device)

        # Iteration weighting (strategy net only, per Deep CFR paper)
        if use_iteration_weight:
            weights = iterations.to(self.device)
            weights = weights / weights.sum()
        else:
            weights = None

        n = len(encodings)
        total_loss = 0.0

        net.train()
        for epoch in range(epochs):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                x = encodings[idx]
                y = targets[idx]

                pred = net(x)

                if is_strategy:
                    log_pred = F.log_softmax(pred, dim=-1)
                    loss_per = -(y * log_pred).sum(dim=-1)
                else:
                    loss_per = ((pred - y) ** 2).sum(dim=-1)

                if weights is not None:
                    w = weights[idx]
                    w = w / w.sum()
                    loss = (loss_per * w).sum()
                else:
                    loss = loss_per.mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            total_loss = epoch_loss / max(n_batches, 1)

        net.eval()
        return total_loss


    def train(self, iters=50000, seed=42, retrain_every=500,
              adv_epochs=4, strat_epochs=60, batch_size=512,
              progress=True):
        """
        Run Deep CFR training.

        Key design choices:
        - Advantage net: retrained periodically, few epochs, UNWEIGHTED,
          persistent optimizer (preserves Adam momentum)
        - Strategy net: trained once at the end, iteration-WEIGHTED
        - Large reservoir buffers to retain all experience
        - Hand strength features in encoding so network focuses on strategy
        """
        rng = random.Random(seed)
        deck = make_deck()
        interval = max(1, iters // 20)
        usum = 0.0
        t0 = time.time()

        self.adv_net.eval()

        if progress:
            print(f"Training Deep CFR ({iters:,} iterations)...")
            print(f"Stack: {self.stack} | Blinds: {self.sb}/{self.bb}")
            print(f"Networks: {sum(p.numel() for p in self.adv_net.parameters()):,} params each")
            print()

        for t in range(iters):
            shuffled = list(deck)
            rng.shuffle(shuffled)
            h0, h1 = (shuffled[0], shuffled[1]), (shuffled[2], shuffled[3])
            board = shuffled[4:9]
            hands = [h0, h1]

            init_stacks = [self.stack - self.sb, self.stack - self.bb]
            init_bets = [self.sb, self.bb]

            # Traverse for both players
            usum += self._traverse(
                0, 0, init_stacks, init_bets, 0, '', 0,
                0, hands, board, t + 1, rng
            )
            self._traverse(
                0, 0, init_stacks, init_bets, 0, '', 0,
                1, hands, board, t + 1, rng
            )

            # Periodically retrain advantage network (unweighted, persistent optimizer)
            if (t + 1) % retrain_every == 0 and len(self.adv_buffer) >= batch_size:
                loss = self._train_network(
                    self.adv_net, self.adv_optimizer, self.adv_buffer,
                    epochs=adv_epochs, batch_size=batch_size,
                    use_iteration_weight=False
                )
                self.adv_net.eval()

                if progress:
                    el = time.time() - t0
                    print(f"  [{(t+1)/iters*100:5.1f}%] {t+1:>7,} | "
                          f"EV: {usum/(t+1):+.2f} | "
                          f"adv_loss: {loss:.4f} | "
                          f"buf: {len(self.adv_buffer):,}/{len(self.strat_buffer):,} | "
                          f"{(t+1)/el:.0f}/s")

            elif progress and (t + 1) % interval == 0:
                el = time.time() - t0
                print(f"  [{(t+1)/iters*100:5.1f}%] {t+1:>7,} | "
                      f"EV: {usum/(t+1):+.2f} | "
                      f"buf: {len(self.adv_buffer):,}/{len(self.strat_buffer):,} | "
                      f"{(t+1)/el:.0f}/s")

        # Final: train strategy network (iteration-weighted)
        if progress:
            print(f"\n  Training strategy network on {len(self.strat_buffer):,} samples...")

        self.strat_optimizer = torch.optim.Adam(self.strat_net.parameters(), lr=1e-3)
        strat_loss = self._train_network(
            self.strat_net, self.strat_optimizer, self.strat_buffer,
            epochs=strat_epochs, batch_size=batch_size,
            use_iteration_weight=True, is_strategy=True
        )

        elapsed = time.time() - t0
        if progress:
            print(f"  Strategy loss: {strat_loss:.4f}")
            print(f"\nDone in {elapsed:.1f}s.\n")


    def save(self, path='model.pt'):
        """Save trained networks and config to disk."""
        torch.save({
            'adv_net': self.adv_net.state_dict(),
            'strat_net': self.strat_net.state_dict(),
            'stack': self.stack,
            'sb': self.sb,
            'bb': self.bb,
        }, path)
        print(f"  Model saved to {path}")

    @classmethod
    def load(cls, path='model.pt', device='cpu'):
        """Load a previously trained bot from disk."""
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        bot = cls(
            stack=checkpoint['stack'],
            sb=checkpoint['sb'],
            bb=checkpoint['bb'],
            device=device,
        )
        bot.adv_net.load_state_dict(checkpoint['adv_net'])
        bot.strat_net.load_state_dict(checkpoint['strat_net'])
        bot.adv_net.eval()
        bot.strat_net.eval()
        print(f"  Model loaded from {path}")
        return bot


    @torch.no_grad()
    def get_action(self, hole_cards, board, street, pot, to_call,
                   facing_bet, n_bets, position, rng=None):
        """
        Choose an action using the trained strategy network.

        Args:
            hole_cards: tuple of 2 cards
            board: list of visible board cards
            street: 0=preflop, 1=flop, 2=turn, 3=river
            pot: total pot in chips
            to_call: chips needed to call (0 if not facing bet)
            facing_bet: True if facing a bet/raise
            n_bets: number of bets/raises so far this street
            position: 1=IP (button), 0=OOP

        Returns (action_char, probabilities, action_list)
        """
        can_raise = n_bets < MAX_BETS

        enc = encode_state(
            hole_cards, board, pot, self.stack,
            street, to_call, n_bets, position
        ).unsqueeze(0).to(self.device)

        logits = self.strat_net(enc).squeeze(0)
        mask = legal_mask(facing_bet, can_raise)

        # Mask illegal actions
        logits[~mask] = float('-inf')
        probs = F.softmax(logits, dim=0)

        # Build legal action list
        char_map = IDX_TO_CHAR_FACING if facing_bet else IDX_TO_CHAR
        legal_indices = mask.nonzero(as_tuple=True)[0].tolist()
        action_list = [char_map[i] for i in legal_indices]
        action_probs = [probs[i].item() for i in legal_indices]

        # Sample
        if rng is None:
            rng = random.Random()

        r = rng.random()
        cum = 0.0
        chosen_char = action_list[-1]
        for i, p in enumerate(action_probs):
            cum += p
            if r < cum:
                chosen_char = action_list[i]
                break

        return chosen_char, action_probs, action_list

    @torch.no_grad()
    def print_strategy_sample(self):
        """Show what the strategy net outputs for sample hands."""
        print("=== Strategy Network Outputs ===\n")

        test_hands = [
            ((14, 3), (14, 0)),   # AA
            ((13, 2), (12, 2)),   # KQs
            ((10, 0), (10, 1)),   # TT
            ((14, 3), (7, 1)),    # A7o
            ((8, 0), (7, 0)),     # 87s
            ((2, 1), (7, 3)),     # 72o
        ]
        hand_labels = ['AA', 'KQs', 'TT', 'A7o', '87s', '72o']

        boards = [
            [],                                          # preflop
            [(14, 2), (10, 1), (3, 0)],                  # A-T-3 flop
            [(7, 0), (6, 1), (2, 2)],                    # 7-6-2 flop
        ]
        board_labels = ['Preflop', 'A-T-3 flop', '7-6-2 flop']

        for bi, board in enumerate(boards):
            street = 0 if not board else 1
            print(f"  {board_labels[bi]}:")
            for hi, hand in enumerate(test_hands):
                enc = encode_state(hand, board, 4, self.stack, street, 0, 0, 1)
                logits = self.strat_net(enc.unsqueeze(0)).squeeze(0)
                mask = legal_mask(False, True)
                logits[~mask] = float('-inf')
                probs = F.softmax(logits, dim=0)
                print(f"    {hand_labels[hi]:>4s}: "
                      f"chk {probs[1]:.2f}  "
                      f"half {probs[2]:.2f}  "
                      f"pot {probs[3]:.2f}")
            print()


# Exploitative Wrapper
class ExploitativeBot:
    """
    Wraps a DeepCFRBot with online opponent modeling and hand evaluation.

    Uses actual hand strength (from hand_eval) combined with opponent
    tendencies to make exploitative adjustments:
    - vs TAG: fold to their bets (they never bluff), steal when they check
    - vs LAG: call with made hands (they over-bluff), bet when they check
      (check = air since they always bet made hands)
    - vs calling station: value bet thin, never bluff
    """

    def __init__(self, base_bot):
        self.base_bot = base_bot
        self.stack = base_bot.stack
        self.reset_stats()

    def reset_stats(self):
        """Reset opponent model for a new opponent."""
        self.opp_bets = 0        # times opponent bet or raised
        self.opp_checks = 0      # times opponent checked
        self.opp_calls = 0       # times opponent called
        self.opp_folds = 0       # times opponent folded
        self.opp_actions = 0     # total opponent decisions
        self.opp_folds_to_bet = 0  # times opponent folded facing our bet
        self.opp_faced_bet = 0   # times opponent faced our bet
        # Track postflop betting patterns separately
        self.opp_postflop_bets = 0
        self.opp_postflop_checks = 0

    def observe_opponent_action(self, action, was_facing_bet, street=0):
        """Track an opponent action to build their profile."""
        self.opp_actions += 1
        if action in ('h', 'b'):
            self.opp_bets += 1
            # Only count voluntary bets (not raises) for check-weakness signal
            if street > 0 and not was_facing_bet:
                self.opp_postflop_bets += 1
        elif action == 'x':
            self.opp_checks += 1
            if street > 0:
                self.opp_postflop_checks += 1
        elif action == 'c':
            self.opp_calls += 1
        elif action == 'f':
            self.opp_folds += 1

        if was_facing_bet:
            self.opp_faced_bet += 1
            if action == 'f':
                self.opp_folds_to_bet += 1

    @property
    def opp_aggression(self):
        """Fraction of opponent actions that are bets/raises."""
        if self.opp_actions < 20:
            return 0.35  # assume balanced until we have data
        return self.opp_bets / self.opp_actions

    @property
    def opp_fold_rate(self):
        """How often opponent folds when facing a bet."""
        if self.opp_faced_bet < 15:
            return 0.40  # assume balanced
        return self.opp_folds_to_bet / self.opp_faced_bet

    @property
    def opp_postflop_bet_freq(self):
        """How often opponent bets (vs checks) postflop when not facing bet."""
        total = self.opp_postflop_bets + self.opp_postflop_checks
        if total < 15:
            return 0.35  # assume balanced
        return self.opp_postflop_bets / total

    def _get_hand_strength(self, hole_cards, board, street):
        """
        Compute hand strength for exploitation decisions.
        Returns (category, has_made_hand, is_strong).
        """
        if street == 0:
            pf = preflop_strength(hole_cards) / 100.0
            has_pair = hole_cards[0][0] == hole_cards[1][0]
            return (pf, has_pair or pf > 0.35, pf > 0.55)

        rank = best_hand(hole_cards, board)
        category = rank[0]  # 0=high card, 1=pair, 2=two pair, ...

        # has_made_hand: at least a pair
        has_made = category >= 1

        # is_strong: top pair+ or better
        is_strong = category >= 2
        if category == 1:
            # Check if it's top pair using hole card
            pair_rank = rank[1]
            hole_ranks = {hole_cards[0][0], hole_cards[1][0]}
            board_max = max(c[0] for c in board)
            if pair_rank in hole_ranks and pair_rank >= board_max:
                is_strong = True

        return (category / 8.0, has_made, is_strong)

    def get_action(self, hole_cards, board, street, pot, to_call,
                   facing_bet, n_bets, position, rng=None):
        """Get action with exploitative adjustments based on hand strength + opponent model."""
        char, probs, actions = self.base_bot.get_action(
            hole_cards, board, street, pot, to_call,
            facing_bet, n_bets, position, rng
        )

        # Need enough data before exploiting
        if self.opp_actions < 30:
            return char, probs, actions

        agg = self.opp_aggression
        fold_rate = self.opp_fold_rate
        postflop_bet_freq = self.opp_postflop_bet_freq
        strength, has_made, is_strong = self._get_hand_strength(hole_cards, board, street)

        # Detect opponent archetype
        is_lag = agg > 0.42
        is_tag = agg < 0.25
        is_station = fold_rate < 0.15

        # For extreme opponent profiles, use direct counter-strategy
        # instead of mild CFR adjustments
        override = self._try_override(
            actions, is_lag, is_tag, is_station, facing_bet, street,
            has_made, is_strong, strength, agg, fold_rate, postflop_bet_freq
        )
        if override is not None:
            new_probs = override
        else:
            # Mild adjustments for moderate opponents
            prob_dict = {a: p for a, p in zip(actions, probs)}
            if facing_bet:
                self._adjust_facing_bet(prob_dict, agg, has_made, is_strong)
            else:
                self._adjust_not_facing(prob_dict, fold_rate, has_made, strength)
            total = sum(prob_dict.values())
            if total > 0:
                for a in prob_dict:
                    prob_dict[a] /= total
            new_probs = [prob_dict[a] for a in actions]

        # Sample from adjusted probabilities
        if rng is None:
            rng = random.Random()
        r = rng.random()
        cum = 0.0
        chosen = actions[-1]
        for i, p in enumerate(new_probs):
            cum += p
            if r < cum:
                chosen = actions[i]
                break

        return chosen, new_probs, actions

    def _try_override(self, actions, is_lag, is_tag, is_station,
                      facing_bet, street, has_made, is_strong,
                      strength, agg, fold_rate, pf_bet_freq):
        """
        For opponents with extreme, clear tendencies, return a direct
        counter-strategy probability distribution (bypassing CFR).
        Returns None if no override applies.
        """
        action_set = set(actions)

        # ========================================
        # Counter-strategy vs LAG (high aggression)
        # ========================================
        if is_lag:
            probs = {a: 0.0 for a in actions}

            if facing_bet:
                if street == 0:
                    # LAG raises ~85% of hands preflop
                    # Call with anything remotely playable (pf > 0.20)
                    # 3-bet with strong hands
                    if strength > 0.55:
                        # Premium: 3-bet
                        if 'b' in action_set:
                            probs['b'] = 0.7
                            probs['c'] = 0.3
                        else:
                            probs['c'] = 1.0
                    elif strength > 0.20:
                        # Playable: call (profitable vs wide range)
                        probs['c'] = 1.0
                    else:
                        # Garbage: fold
                        probs['f'] = 1.0
                else:
                    # Postflop: LAG bets pair+ always, bluffs 40% of air
                    # ~37-44% of their bets are bluffs → calling with pair+ is +EV
                    if is_strong:
                        # Top pair+ or better: raise for value sometimes
                        if 'h' in action_set:
                            probs['c'] = 0.6
                            probs['h'] = 0.4
                        else:
                            probs['c'] = 1.0
                    elif has_made:
                        # Any pair: call (we beat bluffs)
                        probs['c'] = 1.0
                    else:
                        # Air: fold (can't beat value OR better air)
                        probs['f'] = 1.0
            else:
                # Not facing bet (we act first, or LAG checked)
                if street == 0:
                    # We're SB acting first preflop
                    if strength > 0.45:
                        probs['b'] = 0.6 if 'b' in action_set else 0.0
                        probs['h'] = 0.4 if 'h' in action_set else 0.0
                        if not probs['b'] and not probs['h']:
                            probs['x'] = 1.0
                    elif strength > 0.25:
                        probs['h'] = 0.5 if 'h' in action_set else 0.0
                        probs['x'] = 0.5
                    else:
                        probs['x'] = 1.0
                else:
                    # Postflop: LAG bets pair+ 100%. If they checked = AIR.
                    # Bet with everything! They fold 85% of air to bets.
                    if pf_bet_freq > 0.45:
                        # LAG checked → they have air
                        if has_made:
                            # Value bet big
                            probs['b'] = 0.8 if 'b' in action_set else 0.0
                            probs['h'] = 0.2 if 'h' in action_set else 0.0
                        else:
                            # Bluff (they fold 85% of air!)
                            probs['h'] = 0.7 if 'h' in action_set else 0.0
                            probs['x'] = 0.3
                    else:
                        # Not enough data for check-weakness, use fold_rate
                        if fold_rate > 0.40 and has_made:
                            probs['h'] = 0.6 if 'h' in action_set else 0.0
                            probs['b'] = 0.2 if 'b' in action_set else 0.0
                            probs['x'] = 0.2
                        else:
                            return None  # fall back to CFR

            # Normalize
            total = sum(probs.values())
            if total <= 0:
                return None
            return [probs[a] / total for a in actions]

        # ========================================
        # Counter-strategy vs TAG (low aggression)
        # ========================================
        if is_tag:
            probs = {a: 0.0 for a in actions}

            if facing_bet:
                # TAG only bets with real hands. Respect their bets.
                if street == 0:
                    if strength > 0.55:
                        # Premium: 3-bet
                        probs['b'] = 0.5 if 'b' in action_set else 0.0
                        probs['c'] = 0.5
                    elif strength > 0.40:
                        # Good: call
                        probs['c'] = 1.0
                    else:
                        # Weak: fold (TAG only raises strong)
                        probs['f'] = 1.0
                else:
                    # Postflop: TAG bets = strong hand
                    if is_strong:
                        # Top pair+: call (we can compete)
                        probs['c'] = 1.0
                    else:
                        # Below top pair: fold (TAG has us beat)
                        probs['f'] = 0.9
                        probs['c'] = 0.1
            else:
                # Not facing bet: TAG checks = weakness
                # Steal with bets (TAG folds most non-premium hands)
                if street == 0:
                    # Preflop first to act: raise wide to steal
                    if strength > 0.25:
                        probs['h'] = 0.7 if 'h' in action_set else 0.0
                        probs['b'] = 0.3 if 'b' in action_set else 0.0
                    else:
                        probs['h'] = 0.4 if 'h' in action_set else 0.0
                        probs['x'] = 0.6
                else:
                    # Postflop: TAG checked = middle pair or worse
                    # Bet to push them off
                    if has_made:
                        probs['h'] = 0.6 if 'h' in action_set else 0.0
                        probs['b'] = 0.3 if 'b' in action_set else 0.0
                        probs['x'] = 0.1
                    else:
                        # Bluff into TAG's weakness (they fold non-pairs)
                        probs['h'] = 0.5 if 'h' in action_set else 0.0
                        probs['x'] = 0.5

            total = sum(probs.values())
            if total <= 0:
                return None
            return [probs[a] / total for a in actions]

        # ========================================
        # Counter-strategy vs CallStation
        # ========================================
        if is_station:
            probs = {a: 0.0 for a in actions}
            if not facing_bet:
                if has_made:
                    # Value bet (they call with anything)
                    probs['b'] = 0.7 if 'b' in action_set else 0.0
                    probs['h'] = 0.3 if 'h' in action_set else 0.0
                else:
                    # Don't bluff a calling station
                    probs['x'] = 1.0
            else:
                # Facing bet from station is rare (they mostly call/check)
                return None  # fall back to CFR
            total = sum(probs.values())
            if total <= 0:
                return None
            return [probs[a] / total for a in actions]

        return None  # no override for moderate opponents

    def _adjust_facing_bet(self, prob_dict, agg, has_made, is_strong):
        """Mild adjustments for moderate opponents facing a bet."""
        if 'f' not in prob_dict or 'c' not in prob_dict:
            return
        if agg > 0.35 and has_made:
            shift = prob_dict['f'] * 0.3
            prob_dict['f'] -= shift
            prob_dict['c'] += shift
        elif agg < 0.30 and not is_strong:
            shift = prob_dict['c'] * 0.3
            prob_dict['c'] -= shift
            prob_dict['f'] += shift

    def _adjust_not_facing(self, prob_dict, fold_rate, has_made, strength):
        """Mild adjustments for moderate opponents not facing a bet."""
        if 'x' not in prob_dict:
            return
        bet_actions = [a for a in ('h', 'b') if a in prob_dict]
        if not bet_actions:
            return
        if fold_rate > 0.40 and (has_made or fold_rate > 0.55):
            shift = prob_dict['x'] * 0.3
            prob_dict['x'] -= shift
            for ba in bet_actions:
                prob_dict[ba] += shift / len(bet_actions)
