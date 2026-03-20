#!/usr/bin/env python3
"""
Play Heads-Up No-Limit Texas Hold'em Against a Deep CFR Bot
"""

import os
import sys
import random
import argparse
from hand_eval import make_deck, card_str, hand_str, best_hand, hand_name
from bot import DeepCFRBot, MAX_BETS

STREET_NAMES = ['Preflop', 'Flop', 'Turn', 'River']

# ANSI colors
DIM = '\033[2m'
BOLD = '\033[1m'
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
WHITE = '\033[97m'
RESET = '\033[0m'


def colored_card(card):
    """Render a card with red for hearts/diamonds, white for clubs/spades."""
    r, s = card
    RANK_NAMES = {2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'T',11:'J',12:'Q',13:'K',14:'A'}
    SUIT_SYMBOLS = {0:'♣', 1:'♦', 2:'♥', 3:'♠'}
    rank = RANK_NAMES[r]
    suit = SUIT_SYMBOLS[s]
    if s in (1, 2):  # diamonds, hearts
        return f"{RED}{rank}{suit}{RESET}"
    return f"{WHITE}{rank}{suit}{RESET}"


def render_card(card):
    """Card in a box: [A♠]"""
    return f"[{colored_card(card)}]"


def clear_screen():
    print('\033[2J\033[H', end='')


# Game State
class Game:
    """Tracks a single hand of heads-up NLHE."""

    def __init__(self, human_chips, bot_chips, sb=1, bb=2, stack_cap=200):
        self.sb = sb
        self.bb = bb
        self.stack_cap = stack_cap
        self.human_chips = human_chips
        self.bot_chips = bot_chips

    def start_hand(self, human_is_button, hand_num):
        self.hand_num = hand_num
        self.human_is_button = human_is_button

        deck = make_deck()
        random.shuffle(deck)
        self.human_hand = (deck[0], deck[1])
        self.bot_hand = (deck[2], deck[3])
        self.board = deck[4:9]

        if human_is_button:
            self.human_seat = 0
            self.bot_seat = 1
        else:
            self.human_seat = 1
            self.bot_seat = 0

        sb_chips = self.human_chips if human_is_button else self.bot_chips
        bb_chips = self.bot_chips if human_is_button else self.human_chips
        self.stacks = [sb_chips - self.sb, bb_chips - self.bb]
        self.bets = [self.sb, self.bb]
        self.pot = 0
        self.street = 0
        self.cp = 0
        self.street_hist = ''
        self.n_bets = 0
        self.done = False
        self.winner = -1
        self.human_rank = None
        self.bot_rank = None

    def visible_board(self):
        if self.street <= 0: return []
        if self.street == 1: return self.board[:3]
        if self.street == 2: return self.board[:4]
        return self.board[:5]

    def is_human_turn(self):
        return self.cp == self.human_seat

    def total_pot(self):
        return self.pot + self.bets[0] + self.bets[1]

    def to_call(self):
        opp = 1 - self.cp
        return max(0, self.bets[opp] - self.bets[self.cp])

    def facing_bet(self):
        return self.to_call() > 0

    def human_stack(self):
        return self.stacks[self.human_seat]

    def bot_stack(self):
        return self.stacks[self.bot_seat]

    def bet_amount(self, action):
        cp = self.cp
        opp = 1 - cp
        tc = self.to_call()
        cur_pot = self.total_pot()

        if action in ('f', 'x'):
            return 0
        elif action == 'c':
            return min(tc, self.stacks[cp])
        elif action == 'h':
            pot_after = cur_pot + tc
            size = max(pot_after // 2, self.bb)
            return min(tc + size, self.stacks[cp])
        elif action == 'b':
            pot_after = cur_pot + tc
            size = max(pot_after, self.bb)
            return min(tc + size, self.stacks[cp])
        return 0

    def legal_actions(self):
        facing = self.facing_bet()
        can_raise = self.n_bets < MAX_BETS and self.stacks[self.cp] > self.to_call()

        if not facing:
            return ['x', 'h', 'b']
        a = ['f', 'c']
        if can_raise:
            a.extend(['h', 'b'])
        return a

    def apply(self, action):
        cp = self.cp
        opp = 1 - cp
        tc = self.to_call()

        self.street_hist += action

        if action == 'f':
            self.done = True
            self.winner = opp
            return

        if action == 'x':
            if len(self.street_hist) >= 2 and self.street_hist[-2] == 'x':
                self._next_street()
            else:
                self.cp = opp
            return

        if action == 'c':
            amt = min(tc, self.stacks[cp])
            self.bets[cp] += amt
            self.stacks[cp] -= amt

            if self.street == 0 and self.street_hist == 'c':
                self.cp = opp
                return
            self._next_street()
            return

        amt = self.bet_amount(action)
        self.bets[cp] += amt
        self.stacks[cp] -= amt
        self.n_bets += 1
        self.cp = opp

    def _next_street(self):
        self.pot += self.bets[0] + self.bets[1]
        self.bets = [0, 0]
        self.street += 1
        self.street_hist = ''
        self.n_bets = 0

        if self.street > 3 or self.stacks[0] <= 0 or self.stacks[1] <= 0:
            self._showdown()
            return

        self.cp = 1

    def _showdown(self):
        self.done = True
        self.human_rank = best_hand(self.human_hand, self.board)
        self.bot_rank = best_hand(self.bot_hand, self.board)

        if self.human_rank > self.bot_rank:
            self.winner = self.human_seat
        elif self.bot_rank > self.human_rank:
            self.winner = self.bot_seat
        else:
            self.winner = 2

    def settle(self):
        total = self.total_pot()
        if self.winner == 2:
            half = total / 2.0
            win0 = self.stacks[0] + half
            win1 = self.stacks[1] + half
        elif self.winner == 0:
            win0 = self.stacks[0] + total
            win1 = self.stacks[1]
        else:
            win0 = self.stacks[0]
            win1 = self.stacks[1] + total

        old = self.human_chips
        if self.human_is_button:
            self.human_chips = win0
            self.bot_chips = win1
        else:
            self.human_chips = win1
            self.bot_chips = win0
        return self.human_chips - old


# Display
def show_state(g: Game):
    clear_screen()
    pos = "BTN" if g.human_is_button else "BB"
    street = STREET_NAMES[g.street]

    # Header
    print(f"\n {DIM}Hand #{g.hand_num}{RESET}    {BOLD}{street}{RESET}    {DIM}You: {pos}{RESET}")
    print(f" {DIM}{'─' * 44}{RESET}")

    # Bot info (top of "table")
    print(f"\n {DIM}Bot{RESET}  {YELLOW}{g.bot_stack()}{RESET} chips")

    # Board (center of "table")
    board = g.visible_board()
    if board:
        cards = '  '.join(render_card(c) for c in board)
    else:
        cards = f"{DIM}·  ·  ·  ·  ·{RESET}"
    print(f"\n   {cards}")
    print(f"   {BOLD}Pot: {YELLOW}{g.total_pot()}{RESET}")

    if g.bets[0] > 0 or g.bets[1] > 0:
        hb = g.bets[g.human_seat]
        bb = g.bets[g.bot_seat]
        parts = []
        if hb > 0:
            parts.append(f"you: {hb}")
        if bb > 0:
            parts.append(f"bot: {bb}")
        print(f"   {DIM}bets  {' / '.join(parts)}{RESET}")

    # Your hand (bottom of "table")
    h = g.human_hand
    print(f"\n {DIM}You{RESET}  {YELLOW}{g.human_stack()}{RESET} chips")
    print(f" Hand: {render_card(h[0])}  {render_card(h[1])}")
    print()


def show_menu(g: Game):
    actions = g.legal_actions()
    for i, a in enumerate(actions):
        amt = g.bet_amount(a)
        tc = g.to_call()
        facing = g.facing_bet()

        if a == 'x':
            label = "Check"
        elif a == 'f':
            label = "Fold"
        elif a == 'c':
            label = f"Call {tc}"
        elif a == 'h':
            if facing:
                label = f"Raise to {g.bets[g.cp] + amt} {DIM}(½ pot){RESET}"
            else:
                label = f"Bet {amt} {DIM}(½ pot){RESET}"
        elif a == 'b':
            if facing:
                label = f"Raise to {g.bets[g.cp] + amt} {DIM}(pot){RESET}"
            else:
                label = f"Bet {amt} {DIM}(pot){RESET}"
        else:
            label = a

        print(f"  {CYAN}{i+1}{RESET}  {label}")
    print()
    return actions


def show_bot_action(g: Game, action):
    amt = g.bet_amount(action)
    tc = g.to_call()
    facing = g.facing_bet()

    if action == 'f':
        msg = "Bot folds"
    elif action == 'x':
        msg = "Bot checks"
    elif action == 'c':
        msg = f"Bot calls {tc}"
    elif action == 'h':
        if facing:
            msg = f"Bot raises to {g.bets[g.cp] + amt}"
        else:
            msg = f"Bot bets {amt}"
    elif action == 'b':
        if facing:
            msg = f"Bot raises to {g.bets[g.cp] + amt}"
        else:
            msg = f"Bot bets {amt}"
    else:
        msg = f"Bot: {action}"

    print(f" {DIM}>> {msg}{RESET}")
    input(f" {DIM}(press enter){RESET}")


def show_result(g: Game, net):
    clear_screen()
    print()

    if g.human_rank is not None:
        print(f" {BOLD}SHOWDOWN{RESET}\n")
        board_str = '  '.join(render_card(c) for c in g.board)
        print(f"   {board_str}\n")
        print(f" You:  {render_card(g.human_hand[0])}  {render_card(g.human_hand[1])}  {DIM}{hand_name(g.human_rank)}{RESET}")
        print(f" Bot:  {render_card(g.bot_hand[0])}  {render_card(g.bot_hand[1])}  {DIM}{hand_name(g.bot_rank)}{RESET}")
    else:
        b = g.bot_hand
        print(f" Bot mucked: {render_card(b[0])}  {render_card(b[1])}")

    print()
    if net > 0:
        print(f" {GREEN}{BOLD}+{net:.0f}{RESET} {GREEN}chips{RESET}")
    elif net < 0:
        print(f" {RED}{BOLD}{net:.0f}{RESET} {RED}chips{RESET}")
    else:
        print(f" {YELLOW}Split pot{RESET}")

    print(f"\n {DIM}Chips  You: {g.human_chips:.0f}  |  Bot: {g.bot_chips:.0f}{RESET}")
    print(f" {DIM}{'─' * 44}{RESET}")


# Main Loop
def play_hand(g: Game, bot: DeepCFRBot, bot_rng):
    while not g.done:
        show_state(g)

        if g.is_human_turn():
            actions = show_menu(g)
            action = get_input(actions)
            if action is None:
                continue
            g.apply(action)
        else:
            vis = g.visible_board()
            a, probs, acts = bot.get_action(
                g.bot_hand, vis, g.street, g.total_pot(),
                g.to_call(), g.facing_bet(), g.n_bets,
                1 if g.bot_seat == 0 else 0,
                bot_rng
            )
            show_bot_action(g, a)
            g.apply(a)

    net = g.settle()
    show_result(g, net)


def get_input(actions):
    try:
        raw = input(f" {BOLD}>{RESET} ").strip()
        idx = int(raw) - 1
        if 0 <= idx < len(actions):
            return actions[idx]
    except (ValueError, EOFError, KeyboardInterrupt):
        pass
    print(f" {DIM}Pick a number{RESET}")
    return None


def main():
    parser = argparse.ArgumentParser(description='Play Texas Hold\'em vs Deep CFR Bot')
    parser.add_argument('--iters', type=int, default=30000,
                        help='Training iterations (default: 30000)')
    parser.add_argument('--stack', type=int, default=200,
                        help='Starting stack (default: 200 chips)')
    parser.add_argument('--sb', type=int, default=1, help='Small blind')
    parser.add_argument('--bb', type=int, default=2, help='Big blind')
    parser.add_argument('--seed', type=int, default=42, help='Training seed')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: cpu or cuda (default: cpu)')
    parser.add_argument('--model', type=str, default='model.pt',
                        help='Model file path (default: model.pt)')
    parser.add_argument('--retrain', action='store_true',
                        help='Force retrain even if saved model exists')
    args = parser.parse_args()

    if os.path.exists(args.model) and not args.retrain:
        bot = DeepCFRBot.load(args.model, device=args.device)
        bot.stack = args.stack
    else:
        print(f"\n Training bot ({args.iters:,} iterations)...\n")
        bot = DeepCFRBot(stack=args.stack, sb=args.sb, bb=args.bb, device=args.device)
        bot.train(iters=args.iters, seed=args.seed)
        bot.save(args.model)

    bot_rng = random.Random(777)
    g = Game(args.stack, args.stack, args.sb, args.bb, args.stack)

    clear_screen()
    print(f"\n {BOLD}Heads-Up No-Limit Hold'em{RESET}")
    print(f" {DIM}{args.stack} chips  |  blinds {args.sb}/{args.bb}{RESET}")
    print(f" {DIM}{'─' * 44}{RESET}")
    input(f"\n {DIM}Press enter to start{RESET}")

    hand_num = 0
    while True:
        if g.human_chips < args.bb:
            clear_screen()
            print(f"\n {RED}Busted. Game over.{RESET}\n")
            break
        if g.bot_chips < args.bb:
            clear_screen()
            print(f"\n {GREEN}Bot is busted! You win the match.{RESET}\n")
            break

        hand_num += 1
        g.start_hand(human_is_button=(hand_num % 2 == 1), hand_num=hand_num)
        play_hand(g, bot, bot_rng)

        print()
        try:
            ans = input(f" {DIM}Next hand? [Y/n]{RESET} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = 'n'
        if ans == 'n':
            break

    net = g.human_chips - args.stack
    if net >= 0:
        color = GREEN
    else:
        color = RED
    print(f" {hand_num} hands, {color}{'+' if net >= 0 else ''}{net:.0f} chips{RESET}\n")


if __name__ == '__main__':
    main()
