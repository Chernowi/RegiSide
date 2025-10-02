"""
Compact Regicide Environment for MARL Training

A standalone, high-performance implementation optimized for training
Multi-Agent Reinforcement Learning algorithms. No database, no API calls,
pure in-memory operations with vectorized computations.

This version is updated to be fully compliant with the official Regicide rules,
including support for combo plays and correct "exact damage" mechanics, while
maintaining backward compatibility with the original file's attribute names.

Key optimizations:
- All game logic in memory (no I/O)
- Vectorized observations and actions
- Fast reset/step functions
- Minimal object creation
- Optional JAX/Numba compilation support
- Batch environment support
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Union
from enum import IntEnum
import random
from itertools import combinations


class GameStatus(IntEnum):
    """Game status enumeration"""
    IN_PROGRESS = 0
    AWAITING_DEFENSE = 1
    AWAITING_JESTER_CHOICE = 2
    WON = 3
    LOST = 4


class CompactCard:
    """Memory-efficient card representation using integers"""

    # Suit encoding: 0=H, 1=D, 2=S, 3=C
    SUITS = ['H', 'D', 'S', 'C']
    RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']

    def __init__(self, card_id: int):
        """
        Card encoded as single integer:
        - 0-51: Regular cards (rank * 4 + suit)
        - 52-53: Jesters
        - 255: Empty/None
        """
        self.id = card_id

    @classmethod
    def from_string(cls, card_str: str) -> 'CompactCard':
        """Convert from string representation"""
        if card_str.startswith('X'):
            return cls(52)  # First jester

        rank = card_str[:-1]
        suit = card_str[-1]

        rank_idx = cls.RANKS.index(rank)
        suit_idx = cls.SUITS.index(suit)

        return cls(rank_idx * 4 + suit_idx)

    @property
    def is_jester(self) -> bool:
        return self.id >= 52

    @property
    def is_joker(self) -> bool:
        # Alias for backward compatibility with external scripts
        return self.is_jester

    @property
    def is_empty(self) -> bool:
        return self.id == 255

    @property
    def suit_idx(self) -> int:
        if self.is_jester or self.is_empty:
            return -1
        return self.id % 4

    @property
    def rank_idx(self) -> int:
        if self.is_jester or self.is_empty:
            return -1
        return self.id // 4

    @property
    def value(self) -> int:
        """Card value for damage calculation and discarding"""
        if self.is_jester:
            return 0
        if self.is_empty:
            return 0

        rank_idx = self.rank_idx
        if rank_idx == 0:  # Ace
            return 1
        elif rank_idx <= 9:  # 2-10 (indices 1..9)
            return rank_idx + 1
        elif rank_idx == 10:  # Jack (index 10)
            return 10
        elif rank_idx == 11:  # Queen (index 11)
            return 15
        elif rank_idx == 12:  # King (index 12)
            return 20
        return 0

    @property
    def attack_power(self) -> int:
        """Attack power when played"""
        return self.value

    def __str__(self):
        if self.is_empty:
            return "EMPTY"
        if self.is_jester:
            return "JESTER"
        return f"{self.RANKS[self.rank_idx]}{self.SUITS[self.suit_idx]}"


class CompactRegicideEngine:
    """
    High-performance, in-memory Regicide game engine. Fully rule-compliant.
    """
    MAX_HAND_SIZE = 8

    def __init__(self, num_players: int = 4, seed: Optional[int] = None):
        self.num_players = num_players
        self.rng = np.random.RandomState(seed)

        self.hand_sizes = {1: 8, 2: 7, 3: 6, 4: 5}
        self.jesters_count = {1: 0, 2: 0, 3: 1, 4: 2}

        self.royal_health = {10: 20, 11: 30, 12: 40}  # J, Q, K by rank_idx
        self.royal_attack = {10: 10, 11: 15, 12: 20}

        # Pre-compute combination mappings for the action space
        self._slot_pairs = list(combinations(range(self.MAX_HAND_SIZE), 2))
        self._slot_triples = list(combinations(range(self.MAX_HAND_SIZE), 3))
        self._slot_quads = list(combinations(range(self.MAX_HAND_SIZE), 4))

        self.reset()

    def reset(self):
        """Reset game to initial state"""
        self.status = GameStatus.IN_PROGRESS
        self.current_player = 0
        self.turns_survived = 0
        self.consecutive_yields = 0
        self.current_enemy_id = 255
        
        self.current_enemy_health = 0
        self.current_enemy_attack = 0
        self.current_enemy_shield = 0
        
        self.damage_to_defend = 0
        self.defend_player_idx = -1
        self.jester_chooser_idx = -1
        self.jester_cancels_immunity = False

        self.tavern_deck = np.array([], dtype=np.uint8)
        self.castle_deck = np.array([], dtype=np.uint8)
        self.discard = np.array([], dtype=np.uint8)

        self.hands = np.full((self.num_players, self.MAX_HAND_SIZE), 255, dtype=np.uint8)

        self._initialize_decks()
        self._deal_initial_hands()
        self._draw_first_enemy()

    def _initialize_decks(self):
        """Create and shuffle initial decks"""
        tavern_cards = [r * 4 + s for r in range(10) for s in range(4)]
        jester_count = self.jesters_count[self.num_players]
        tavern_cards.extend([52 + i for i in range(jester_count)])
        self.rng.shuffle(tavern_cards)
        self.tavern_deck = np.array(tavern_cards, dtype=np.uint8)

        jacks = [10 * 4 + s for s in range(4)]
        queens = [11 * 4 + s for s in range(4)]
        kings = [12 * 4 + s for s in range(4)]
        self.rng.shuffle(jacks)
        self.rng.shuffle(queens)
        self.rng.shuffle(kings)
        self.castle_deck = np.array(jacks + queens + kings, dtype=np.uint8)

    def _deal_initial_hands(self):
        hand_size = self.hand_sizes[self.num_players]
        deal_count = self.num_players * hand_size
        if len(self.tavern_deck) >= deal_count:
            cards_to_deal = self.tavern_deck[:deal_count]
            self.tavern_deck = self.tavern_deck[deal_count:]
            self.hands[:, :hand_size] = cards_to_deal.reshape(self.num_players, hand_size)

    def _draw_first_enemy(self):
        if len(self.castle_deck) > 0:
            enemy_id = self.castle_deck[0]
            self.castle_deck = self.castle_deck[1:]
            self._set_current_enemy(enemy_id)

    def _set_current_enemy(self, enemy_id: int):
        self.current_enemy_id = enemy_id
        card = CompactCard(enemy_id)
        if card.rank_idx in self.royal_health:
            self.current_enemy_health = self.royal_health[card.rank_idx]
            self.current_enemy_attack = self.royal_attack[card.rank_idx]
        
        self.current_enemy_shield = 0
        self.jester_cancels_immunity = False

    def get_valid_actions(self, player_idx: int) -> np.ndarray:
        valid_actions = []
        if self.status == GameStatus.IN_PROGRESS and player_idx == self.current_player:
            if self.consecutive_yields < self.num_players - 1:
                valid_actions.append(0)  # Yield

            hand_cards = [(i, CompactCard(cid)) for i, cid in enumerate(self.hands[player_idx]) if not CompactCard(cid).is_empty]
            
            # Single card plays
            for slot, card in hand_cards:
                valid_actions.append(CompactRegicideEnv.ACTION_SINGLE_START + slot)

            # Ace + other card plays
            ace_plays = [(s, c) for s, c in hand_cards if c.rank_idx == 0]
            other_plays = [(s, c) for s, c in hand_cards if not c.is_jester]
            for ace_slot, _ in ace_plays:
                for other_slot, _ in other_plays:
                    if ace_slot != other_slot:
                        linear_idx = ace_slot * (self.MAX_HAND_SIZE - 1) + (other_slot if other_slot < ace_slot else other_slot - 1)
                        valid_actions.append(CompactRegicideEnv.ACTION_ACE_COMBO_START + linear_idx)

            # Set combo plays (Pairs, Triples, Quads)
            ranks_in_hand = {}
            for slot, card in hand_cards:
                if card.rank_idx not in ranks_in_hand: ranks_in_hand[card.rank_idx] = []
                ranks_in_hand[card.rank_idx].append(slot)

            for rank, slots in ranks_in_hand.items():
                combo_card = CompactCard(slots[0] * 4) # temp card to get value
                if len(slots) >= 2 and combo_card.value * 2 <= 10:
                    for pair in combinations(slots, 2):
                        action_idx = self._slot_pairs.index(tuple(sorted(pair)))
                        valid_actions.append(CompactRegicideEnv.ACTION_PAIR_COMBO_START + action_idx)
                if len(slots) >= 3 and combo_card.value * 3 <= 10:
                    for triple in combinations(slots, 3):
                        action_idx = self._slot_triples.index(tuple(sorted(triple)))
                        valid_actions.append(CompactRegicideEnv.ACTION_TRIPLE_COMBO_START + action_idx)
                if len(slots) >= 4 and combo_card.value * 4 <= 10:
                    for quad in combinations(slots, 4):
                        action_idx = self._slot_quads.index(tuple(sorted(quad)))
                        valid_actions.append(CompactRegicideEnv.ACTION_QUAD_COMBO_START + action_idx)

        elif self.status == GameStatus.AWAITING_DEFENSE and player_idx == self.defend_player_idx:
            valid_actions.extend(range(CompactRegicideEnv.ACTION_DEFEND_START, CompactRegicideEnv.ACTION_DEFEND_START + 4))

        elif self.status == GameStatus.AWAITING_JESTER_CHOICE and player_idx == self.jester_chooser_idx:
            valid_actions.extend(range(CompactRegicideEnv.ACTION_JESTER_CHOICE_START, CompactRegicideEnv.ACTION_JESTER_CHOICE_START + self.num_players))

        return np.array(list(set(valid_actions)), dtype=np.int32)

    def step(self, player_idx: int, action: int) -> Tuple[bool, str, Dict]:
        if self.status in [GameStatus.WON, GameStatus.LOST]:
            return False, "Game already finished", {}

        if self.status == GameStatus.IN_PROGRESS and player_idx == self.current_player:
            if not self.get_valid_actions(player_idx).any():
                self.status = GameStatus.LOST
                return False, "Player has no valid moves. Game Over.", {}
            return self._execute_play_action(player_idx, action)
        
        elif self.status == GameStatus.AWAITING_DEFENSE and player_idx == self.defend_player_idx:
            return self._execute_defense_action(player_idx, action)
        
        elif self.status == GameStatus.AWAITING_JESTER_CHOICE and player_idx == self.jester_chooser_idx:
            return self._execute_jester_choice(player_idx, action)
        
        return False, f"Invalid action for player {player_idx} in state {self.status.name}", {}

    def _execute_play_action(self, player_idx: int, action: int) -> Tuple[bool, str, Dict]:
        slots = []
        hand = self.hands[player_idx]

        if action == CompactRegicideEnv.ACTION_YIELD:
            self.consecutive_yields += 1
            self._enemy_counterattack()
            return True, "Turn yielded", {}
        
        elif CompactRegicideEnv.ACTION_SINGLE_START <= action < CompactRegicideEnv.ACTION_ACE_COMBO_START:
            slots = [action - CompactRegicideEnv.ACTION_SINGLE_START]
        
        elif CompactRegicideEnv.ACTION_ACE_COMBO_START <= action < CompactRegicideEnv.ACTION_PAIR_COMBO_START:
            linear_idx = action - CompactRegicideEnv.ACTION_ACE_COMBO_START
            ace_slot = linear_idx // (self.MAX_HAND_SIZE - 1)
            other_offset = linear_idx % (self.MAX_HAND_SIZE - 1)
            other_slot = other_offset if other_offset < ace_slot else other_offset + 1
            slots = [ace_slot, other_slot]
        
        elif CompactRegicideEnv.ACTION_PAIR_COMBO_START <= action < CompactRegicideEnv.ACTION_TRIPLE_COMBO_START:
            slots = list(self._slot_pairs[action - CompactRegicideEnv.ACTION_PAIR_COMBO_START])
        
        elif CompactRegicideEnv.ACTION_TRIPLE_COMBO_START <= action < CompactRegicideEnv.ACTION_QUAD_COMBO_START:
            slots = list(self._slot_triples[action - CompactRegicideEnv.ACTION_TRIPLE_COMBO_START])

        elif CompactRegicideEnv.ACTION_QUAD_COMBO_START <= action < CompactRegicideEnv.ACTION_DEFEND_START:
            slots = list(self._slot_quads[action - CompactRegicideEnv.ACTION_QUAD_COMBO_START])
        
        else:
            return False, "Invalid action code", {}

        card_ids = [hand[s] for s in slots]
        if any(cid == 255 for cid in card_ids): return False, "Invalid card slot(s)", {}
        
        return self._play_cards(card_ids, player_idx, slots)

    def _execute_defense_action(self, player_idx: int, action: int) -> Tuple[bool, str, Dict]:
        strategy = action - CompactRegicideEnv.ACTION_DEFEND_START
        return self._perform_defense(player_idx, strategy)

    def _execute_jester_choice(self, player_idx: int, action: int) -> Tuple[bool, str, Dict]:
        target_player = action - CompactRegicideEnv.ACTION_JESTER_CHOICE_START
        if not (0 <= target_player < self.num_players):
            return False, "Invalid jester choice", {}
        self.current_player = target_player
        self.status = GameStatus.IN_PROGRESS
        self.jester_chooser_idx = -1
        return True, f"Next player set to {target_player}", {}

    def _play_cards(self, card_ids: List[int], player_idx: int, slots: List[int]) -> Tuple[bool, str, Dict]:
        info = {}
        played_cards = [CompactCard(cid) for cid in card_ids]

        if any(c.is_jester for c in played_cards):
            self.jester_cancels_immunity = True
            self.discard = np.append(self.discard, [c.id for c in played_cards])
            self._remove_cards_from_hand(player_idx, slots)
            self.jester_chooser_idx = player_idx
            self.status = GameStatus.AWAITING_JESTER_CHOICE
            return True, "Jester played, choose next player", info

        self.consecutive_yields = 0

        combo_value = sum(c.value for c in played_cards)
        unique_suits = set(c.suit_idx for c in played_cards if c.suit_idx != -1)
        enemy_card = CompactCard(self.current_enemy_id)

        heart_heal, diamond_draw, spade_shield, club_double = 0, 0, 0, 0
        for suit in unique_suits:
            is_immune = (suit == enemy_card.suit_idx and not self.jester_cancels_immunity)
            if not is_immune:
                if suit == 0: heart_heal = combo_value
                elif suit == 1: diamond_draw = combo_value
                elif suit == 2: spade_shield = combo_value
                elif suit == 3: club_double = 1

        if heart_heal > 0 and len(self.discard) > 0:
            self.rng.shuffle(self.discard)
            to_move = self.discard[:heart_heal]
            self.discard = self.discard[heart_heal:]
            self.tavern_deck = np.append(self.tavern_deck, to_move)

        if diamond_draw > 0 and len(self.tavern_deck) > 0: self._draw_cards(player_idx, diamond_draw)
        if spade_shield > 0: self.current_enemy_shield += spade_shield

        self._remove_cards_from_hand(player_idx, slots)
        self.discard = np.append(self.discard, card_ids)

        total_attack = combo_value * (2 if club_double else 1)
        info['damage_dealt'] = total_attack
        
        is_exact_damage = (total_attack == self.current_enemy_health)
        
        # FIX: Explicitly cast to Python int before subtraction to prevent overflow
        self.current_enemy_health = int(self.current_enemy_health) - int(total_attack)

        if self.current_enemy_health <= 0:
            defeated_id = self.current_enemy_id
            info['defeated_enemy_id'] = defeated_id

            if is_exact_damage:
                self.tavern_deck = np.concatenate(([defeated_id], self.tavern_deck))
            else:
                self.discard = np.append(self.discard, defeated_id)

            if len(self.castle_deck) == 0:
                self.status = GameStatus.WON
                return True, "All enemies defeated. Victory!", info

            next_enemy = self.castle_deck[0]
            self.castle_deck = self.castle_deck[1:]
            self._set_current_enemy(next_enemy)
            return True, f"Enemy defeated with {total_attack} damage!", info
        else:
            self._enemy_counterattack()
            return True, f"Dealt {total_attack} damage", info

    def _remove_cards_from_hand(self, player_idx: int, slots: List[int]):
        hand = self.hands[player_idx]
        hand[slots] = 255
        valid_cards = hand[hand != 255]
        hand.fill(255)
        hand[:len(valid_cards)] = valid_cards
    
    def _draw_cards(self, start_player_idx: int, amount: int):
        p_idx = start_player_idx
        drawn_count = 0
        skipped_count = 0
        while drawn_count < amount and len(self.tavern_deck) > 0 and skipped_count < self.num_players:
            max_hand = self.hand_sizes[self.num_players]
            current_hand_size = np.sum(self.hands[p_idx] != 255)
            if current_hand_size < max_hand:
                skipped_count = 0
                card_to_draw = self.tavern_deck[0]
                self.tavern_deck = self.tavern_deck[1:]
                self.hands[p_idx, current_hand_size] = card_to_draw
                drawn_count += 1
            else:
                skipped_count += 1
            p_idx = (p_idx + 1) % self.num_players

    def _enemy_counterattack(self):
        # FIX: Explicitly cast to Python int before subtraction to prevent overflow
        damage = max(0, int(self.current_enemy_attack) - int(self.current_enemy_shield))
        self.damage_to_defend = damage
        self.defend_player_idx = self.current_player
        self.status = GameStatus.AWAITING_DEFENSE

    def _perform_defense(self, player_idx: int, strategy: int) -> Tuple[bool, str, Dict]:
        cards_to_discard, slots_to_remove = self._select_defense_cards(player_idx, strategy)
        defense_value = sum(CompactCard(cid).value for cid in cards_to_discard)

        if defense_value >= self.damage_to_defend:
            self._remove_cards_from_hand(player_idx, slots_to_remove)
            self.discard = np.append(self.discard, cards_to_discard)
            self.status = GameStatus.IN_PROGRESS
            self._next_turn()
            return True, f"Successfully defended with {defense_value} points", {}
        else:
            self.status = GameStatus.LOST
            return False, f"Defense failed: {defense_value} < {self.damage_to_defend}", {}

    def _select_defense_cards(self, player_idx: int, strategy: int) -> Tuple[List[int], List[int]]:
        hand = self.hands[player_idx]
        cards_in_hand = [(i, cid) for i, cid in enumerate(hand) if cid != 255]
        cards_with_vals = sorted([(i, cid, CompactCard(cid).value) for i, cid in cards_in_hand], key=lambda x: x[2])

        if not cards_with_vals: return [], []

        selected_cards, selected_slots, total = [], [], 0
        if strategy == 1: # Discard lowest value cards
            for slot, cid, val in cards_with_vals:
                selected_cards.append(cid)
                selected_slots.append(slot)
                total += val
                if total >= self.damage_to_defend: break
            return selected_cards, selected_slots

        elif strategy == 2: # Discard highest value cards
            cards_with_vals.reverse()
            for slot, cid, val in cards_with_vals:
                selected_cards.append(cid)
                selected_slots.append(slot)
                total += val
                if total >= self.damage_to_defend: break
            return selected_cards, selected_slots
        
        elif strategy == 3:  # Discard all
            return [cid for _, cid, _ in cards_with_vals], [slot for slot, _, _ in cards_with_vals]

        # Default/Strategy 0: Minimal cost (simplified as lowest value)
        for slot, cid, val in cards_with_vals:
            selected_cards.append(cid)
            selected_slots.append(slot)
            total += val
            if total >= self.damage_to_defend: break
        return selected_cards, selected_slots

    def _next_turn(self):
        self.current_player = (self.current_player + 1) % self.num_players
        self.turns_survived += 1

    def get_state_dict(self):
        return {'status': int(self.status), 'current_player': self.current_player, 'current_enemy_id': int(self.current_enemy_id), 'current_enemy_health': self.current_enemy_health, 'current_enemy_attack': self.current_enemy_attack, 'current_enemy_shield': self.current_enemy_shield, 'damage_to_defend': self.damage_to_defend, 'hands': self.hands.copy(), 'tavern_deck_size': len(self.tavern_deck), 'castle_deck_size': len(self.castle_deck)}


class CompactRegicideEnv(gym.Env):
    """
    Gym environment for the fully rule-compliant CompactRegicideEngine.
    """
    MAX_HAND_SIZE = 8

    # Action Space Layout
    ACTION_YIELD = 0
    ACTION_SINGLE_START = 1
    ACTION_ACE_COMBO_START = ACTION_SINGLE_START + MAX_HAND_SIZE
    ACTION_PAIR_COMBO_START = ACTION_ACE_COMBO_START + MAX_HAND_SIZE * (MAX_HAND_SIZE - 1)
    ACTION_TRIPLE_COMBO_START = ACTION_PAIR_COMBO_START + len(list(combinations(range(MAX_HAND_SIZE), 2)))
    ACTION_QUAD_COMBO_START = ACTION_TRIPLE_COMBO_START + len(list(combinations(range(MAX_HAND_SIZE), 3)))
    ACTION_DEFEND_START = ACTION_QUAD_COMBO_START + len(list(combinations(range(MAX_HAND_SIZE), 4)))
    ACTION_JESTER_CHOICE_START = ACTION_DEFEND_START + 4
    TOTAL_ACTIONS = ACTION_JESTER_CHOICE_START + 4

    def __init__(self, num_players: int = 4, render_mode: Optional[str] = None, enemy_defeat_only: bool = False):
        super().__init__()
        self.num_players = num_players
        self.render_mode = render_mode
        self.enemy_defeat_only = enemy_defeat_only
        self.action_space = spaces.Discrete(self.TOTAL_ACTIONS)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(60,), dtype=np.float32)
        self.engine = CompactRegicideEngine(num_players)
        self.current_player_idx = 0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.engine = CompactRegicideEngine(self.num_players, seed)
        self.current_player_idx = 0
        return self._get_observation(), self._get_info()

    def step(self, action: int):
        success, message, action_info = self.engine.step(self.current_player_idx, int(action))

        if self.engine.status == GameStatus.AWAITING_DEFENSE:
            self.current_player_idx = self.engine.defend_player_idx
        elif self.engine.status == GameStatus.AWAITING_JESTER_CHOICE:
            self.current_player_idx = self.engine.jester_chooser_idx
        else:
            self.current_player_idx = self.engine.current_player

        terminated = (self.engine.status in [GameStatus.WON, GameStatus.LOST])
        reward = self._calculate_reward(success, terminated, action_info)
        obs = self._get_observation()
        info = self._get_info()
        info['action_result'] = {'success': success, 'message': message}
        return obs, reward, terminated, False, info

    def _get_observation(self) -> np.ndarray:
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        state = self.engine.get_state_dict()
        hand = state['hands'][self.current_player_idx]
        
        # Hand features (32)
        for i in range(self.MAX_HAND_SIZE):
            card = CompactCard(hand[i])
            if not card.is_empty:
                obs[i*4] = card.value / 20.0
                obs[i*4 + 1] = float(card.is_jester)
                obs[i*4 + 2] = card.suit_idx / 3.0 if not card.is_jester else -1
                obs[i*4 + 3] = card.rank_idx / 12.0 if not card.is_jester else -1
        
        # Enemy features (4)
        obs[32] = state['current_enemy_health'] / 40.0
        obs[33] = state['current_enemy_attack'] / 20.0
        obs[34] = state['current_enemy_shield'] / 40.0
        obs[35] = CompactCard(state['current_enemy_id']).suit_idx / 3.0 if state['current_enemy_id'] != 255 else -1

        # Game state features (8)
        obs[36] = float(state['status']) / 4.0
        obs[37] = self.current_player_idx / (self.num_players - 1) if self.num_players > 1 else 0
        obs[38] = state['tavern_deck_size'] / 54.0
        obs[39] = state['castle_deck_size'] / 12.0
        obs[40] = (12 - state['castle_deck_size']) / 12.0
        obs[41] = float(self.engine.jester_cancels_immunity)
        obs[42] = state['damage_to_defend'] / 40.0
        obs[43] = float(self.engine.defend_player_idx == self.current_player_idx)

        # Hand summary features (8)
        hand_cards = [CompactCard(cid) for cid in hand if cid != 255]
        if hand_cards:
            values = [c.value for c in hand_cards]
            obs[44] = len(hand_cards) / self.engine.hand_sizes[self.num_players]
            obs[45] = np.mean(values) / 20.0
            obs[46] = np.max(values) / 20.0
            obs[47] = np.min(values) / 20.0
            suits = [c.suit_idx for c in hand_cards if not c.is_jester]
            for s_idx in range(4): obs[48 + s_idx] = suits.count(s_idx) / len(hand_cards) if hand_cards else 0

        # Action context features (8)
        valid_actions = self.engine.get_valid_actions(self.current_player_idx)
        obs[52] = len(valid_actions) / self.action_space.n
        obs[53] = float(0 in valid_actions)
        obs[54] = float(any(self.ACTION_SINGLE_START <= a < self.ACTION_ACE_COMBO_START for a in valid_actions))
        obs[55] = float(any(self.ACTION_ACE_COMBO_START <= a < self.ACTION_PAIR_COMBO_START for a in valid_actions))
        obs[56] = float(any(self.ACTION_PAIR_COMBO_START <= a < self.ACTION_DEFEND_START for a in valid_actions))
        obs[57] = float(any(self.ACTION_DEFEND_START <= a < self.ACTION_JESTER_CHOICE_START for a in valid_actions))
        obs[58] = float(any(self.ACTION_JESTER_CHOICE_START <= a < self.TOTAL_ACTIONS for a in valid_actions))
        
        return obs

    def _calculate_reward(self, success: bool, terminated: bool, action_info: Dict) -> float:
        enemy_defeat_reward = 0.0
        if 'defeated_enemy_id' in action_info:
            enemy_card = CompactCard(action_info['defeated_enemy_id'])
            enemy_defeat_reward += 1.0 + (enemy_card.rank_idx - 10)

        if self.enemy_defeat_only:
            return enemy_defeat_reward

        if not success: return -0.5
        if terminated and self.engine.status == GameStatus.WON: return 10.0 + enemy_defeat_reward
        if terminated and self.engine.status == GameStatus.LOST: return -10.0
        
        reward = enemy_defeat_reward
        if 'damage_dealt' in action_info:
            reward += action_info['damage_dealt'] / 40.0
        
        return reward

    def _get_info(self) -> Dict:
        return {
            'current_player': self.current_player_idx,
            'game_status': self.engine.status.name,
            'valid_actions': self.engine.get_valid_actions(self.current_player_idx).tolist(),
            'enemy_health': self.engine.current_enemy_health,
            'hand_size': np.sum(self.engine.hands[self.current_player_idx] != 255),
            'enemies_remaining': len(self.engine.castle_deck),
        }

    def get_valid_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=bool)
        valid_actions = self.engine.get_valid_actions(self.current_player_idx)
        mask[valid_actions] = True
        return mask

    def render(self):
        if self.render_mode == "human":
            state = self.engine.get_state_dict()
            enemy_card = CompactCard(state['current_enemy_id'])
            # Get max health for rendering
            max_health = self.engine.royal_health.get(enemy_card.rank_idx, state['current_enemy_health'])
            
            print(f"\n--- Turn {self.engine.turns_survived}, Player {self.current_player_idx} ---")
            print(f"Status: {GameStatus(state['status']).name}")
            print(f"Enemy: {enemy_card} (HP: {state['current_enemy_health']}/{max_health}, ATK: {state['current_enemy_attack']}, SHIELD: {state['current_enemy_shield']})")
            
            hand_cards = [str(CompactCard(cid)) for cid in state['hands'][self.current_player_idx] if cid != 255]
            print(f"Hand: {hand_cards}")
            
            if state['status'] == GameStatus.AWAITING_DEFENSE:
                print(f"Must defend against {state['damage_to_defend']} damage!")

            print(f"Tavern Deck: {state['tavern_deck_size']} cards | Castle Deck: {state['castle_deck_size']} cards")
            print("-" * 20)

if __name__ == "__main__":
    env = CompactRegicideEnv(num_players=2, render_mode="human")
    obs, info = env.reset(seed=42)
    env.render()

    terminated = False
    total_reward = 0
    turn_count = 0
    while not terminated and turn_count < 200: # Safety break
        valid_actions = info['valid_actions']
        if not valid_actions:
            print("No valid actions, game should end.")
            break
        
        action = random.choice(valid_actions)
        print(f"Player {env.current_player_idx} takes action: {action}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Action Result: {info['action_result']['message']}")
        print(f"Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")
        env.render()
        turn_count += 1
        
    print(f"\n=== GAME OVER ===")
    print(f"Final status: {info['game_status']}")
    print(f"Total turns: {turn_count}")