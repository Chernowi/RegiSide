"""
Compact Regicide Environment for MARL Training

A standalone, high-performance implementation optimized for training
Multi-Agent Reinforcement Learning algorithms. No database, no API calls,
pure in-memory operations with vectorized computations.

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


class GameStatus(IntEnum):
    """Game status enumeration"""
    IN_PROGRESS = 0
    AWAITING_DEFENSE = 1 
    AWAITING_JESTER_CHOICE = 2
    WON = 3
    LOST = 4


class CompactCard:
    """Memory-efficient card representation using integers"""
    
    # Suit encoding: 0=H, 1=D, 2=S, 3=C, 4=Joker
    SUITS = ['H', 'D', 'S', 'C']
    RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    
    def __init__(self, card_id: int):
        """
        Card encoded as single integer:
        - 0-51: Regular cards (rank * 4 + suit)  
        - 52-53: Jokers
        - 255: Empty/None
        """
        self.id = card_id
    
    @classmethod
    def from_string(cls, card_str: str) -> 'CompactCard':
        """Convert from string representation"""
        if card_str.startswith('X'):
            return cls(52)  # First joker
        
        rank = card_str[:-1]
        suit = card_str[-1]
        
        rank_idx = cls.RANKS.index(rank)
        suit_idx = cls.SUITS.index(suit)
        
        return cls(rank_idx * 4 + suit_idx)
    
    @property
    def is_joker(self) -> bool:
        return self.id >= 52
    
    @property
    def is_empty(self) -> bool:
        return self.id == 255
    
    @property
    def suit_idx(self) -> int:
        if self.is_joker or self.is_empty:
            return -1
        return self.id % 4
    
    @property
    def rank_idx(self) -> int:
        if self.is_joker or self.is_empty:
            return -1
        return self.id // 4
    
    @property
    def value(self) -> int:
        """Card value for damage calculation"""
        if self.is_joker or self.is_empty:
            return 0
        
        rank_idx = self.rank_idx
        if rank_idx == 0:  # Ace
            return 1
        elif rank_idx <= 8:  # 2-10
            return rank_idx + 1
        elif rank_idx == 9:  # Jack
            return 10  
        elif rank_idx == 10:  # Queen
            return 15
        elif rank_idx == 11:  # King
            return 20
        return 0
    
    @property
    def attack_power(self) -> int:
        """Attack power when played"""
        if self.is_joker or self.is_empty:
            return 0
        
        rank_idx = self.rank_idx
        if rank_idx == 0:  # Ace
            return 1
        elif rank_idx <= 8:  # 2-10  
            return rank_idx + 1
        elif rank_idx == 9:  # Jack
            return 10
        elif rank_idx == 10:  # Queen  
            return 15
        elif rank_idx == 11:  # King
            return 20
        return 0

    def __str__(self):
        if self.is_empty:
            return "EMPTY"
        if self.is_joker:
            return "X"
        return f"{self.RANKS[self.rank_idx]}{self.SUITS[self.suit_idx]}"


class CompactRegicideEngine:
    """
    High-performance, in-memory Regicide game engine.
    
    Uses numpy arrays and vectorized operations for maximum speed.
    No external dependencies, no I/O operations.
    """
    
    def __init__(self, num_players: int = 4, seed: Optional[int] = None):
        self.num_players = num_players
        self.rng = np.random.RandomState(seed)
        
        # Hand sizes by player count
        self.hand_sizes = {1: 8, 2: 7, 3: 6, 4: 5}
        self.jokers_count = {1: 0, 2: 0, 3: 1, 4: 2}
        
        # Royal card stats 
        self.royal_health = {9: 20, 10: 30, 11: 40}  # J, Q, K by rank_idx
        self.royal_attack = {9: 10, 10: 15, 11: 20}
        
        self.reset()
    
    def reset(self):
        """Reset game to initial state"""
        # Game state arrays (vectorized)
        self.status = GameStatus.IN_PROGRESS
        self.current_player = 0
        self.current_enemy_id = 255  # Start with no enemy
        self.current_enemy_health = 0
        self.current_enemy_attack = 0
        self.current_enemy_shield = 0
        self.damage_to_defend = 0
        self.defend_player_idx = -1
        self.jester_chooser_idx = -1
        self.joker_cancels_immunity = False
        
        # Decks as numpy arrays of card IDs
        self.tavern_deck = np.array([], dtype=np.uint8)
        self.castle_deck = np.array([], dtype=np.uint8) 
        self.hospital = np.array([], dtype=np.uint8)
        
        # Player hands (fixed size arrays for efficiency)
        max_hand = self.hand_sizes[self.num_players]
        self.hands = np.full((self.num_players, max_hand), 255, dtype=np.uint8)  # 255 = empty
        
        # Initialize decks and deal
        self._initialize_decks()
        self._deal_initial_hands()
        self._draw_first_enemy()
    
    def _initialize_decks(self):
        """Create and shuffle initial decks"""
        # Create tavern deck (Ace through 10, all suits)
        tavern_cards = []
        for rank_idx in range(9):  # A, 2, 3, 4, 5, 6, 7, 8, 9, 10
            for suit_idx in range(4):
                tavern_cards.append(rank_idx * 4 + suit_idx)
        
        # Add jokers
        joker_count = self.jokers_count[self.num_players]
        for i in range(joker_count):
            tavern_cards.append(52 + i)
        
        # Shuffle
        self.rng.shuffle(tavern_cards)
        self.tavern_deck = np.array(tavern_cards, dtype=np.uint8)
        
        # Create castle deck (Jacks, Queens, Kings)
        castle_cards = []
        for rank_idx in [9, 10, 11]:  # J, Q, K
            for suit_idx in range(4):
                castle_cards.append(rank_idx * 4 + suit_idx)
        
        self.rng.shuffle(castle_cards)
        self.castle_deck = np.array(castle_cards, dtype=np.uint8)
    
    def _deal_initial_hands(self):
        """Deal cards to all players"""
        hand_size = self.hand_sizes[self.num_players]
        
        for player_idx in range(self.num_players):
            for card_slot in range(hand_size):
                if len(self.tavern_deck) > 0:
                    card_id = self.tavern_deck[0]
                    self.tavern_deck = self.tavern_deck[1:]
                    self.hands[player_idx, card_slot] = card_id
    
    def _draw_first_enemy(self):
        """Draw first enemy from castle deck"""
        if len(self.castle_deck) > 0:
            enemy_id = self.castle_deck[0]
            self.castle_deck = self.castle_deck[1:]
            self._set_current_enemy(enemy_id)
    
    def _set_current_enemy(self, enemy_id: int):
        """Set new enemy and its stats"""
        self.current_enemy_id = enemy_id
        card = CompactCard(enemy_id)
        
        if card.rank_idx in self.royal_health:
            self.current_enemy_health = self.royal_health[card.rank_idx]
            self.current_enemy_attack = self.royal_attack[card.rank_idx]
        
        self.current_enemy_shield = 0  # Reset shield
    
    def get_valid_actions(self, player_idx: int) -> np.ndarray:
        """Get valid actions as numpy array of action IDs"""
        valid = []
        
        if self.status == GameStatus.IN_PROGRESS and player_idx == self.current_player:
            # Can always yield (action 0)
            valid.append(0)
            
            # Can play cards from hand
            hand = self.hands[player_idx]
            for slot in range(len(hand)):
                card_id = hand[slot]
                if card_id != 255:  # Not empty
                    # Single card play (actions 1-5)
                    valid.append(1 + slot)
                    
                    # Check for ace companions
                    card = CompactCard(card_id)
                    if card.rank_idx == 0:  # Ace
                        for other_slot in range(len(hand)):
                            if other_slot != slot and hand[other_slot] != 255:
                                other_card = CompactCard(hand[other_slot])
                                if not other_card.is_joker and other_card.rank_idx != 0:
                                    # Ace companion (actions 6-15)
                                    action_id = 6 + slot * 4 + (other_slot if other_slot < slot else other_slot - 1)
                                    if action_id < 16:
                                        valid.append(action_id)
            
            # Check for sets (actions 16-20)
            for rank_idx in range(1, 5):  # ranks 2-5
                count = np.sum((self.hands[player_idx] != 255) & 
                              ((self.hands[player_idx] // 4) == rank_idx))
                if count >= 2:
                    valid.append(16 + rank_idx - 1)
        
        elif self.status == GameStatus.AWAITING_DEFENSE and player_idx == self.defend_player_idx:
            # Defense strategies (actions 26-29)  
            for strategy in range(4):
                valid.append(26 + strategy)
        
        elif self.status == GameStatus.AWAITING_JESTER_CHOICE and player_idx == self.jester_chooser_idx:
            # Player choices (actions 0-3)
            for target_player in range(self.num_players):
                valid.append(target_player)
        
        return np.array(valid, dtype=np.int32)
    
    def step(self, player_idx: int, action: int) -> Tuple[bool, str, Dict]:
        """Execute action and return (success, message, info)"""
        if self.status == GameStatus.WON or self.status == GameStatus.LOST:
            return False, "Game already finished", {}
        
        # Validate it's player's turn
        if self.status == GameStatus.IN_PROGRESS and player_idx != self.current_player:
            return False, f"Not player {player_idx}'s turn", {}
        
        if self.status == GameStatus.AWAITING_DEFENSE and player_idx != self.defend_player_idx:
            return False, f"Player {player_idx} not defending", {}
        
        if self.status == GameStatus.AWAITING_JESTER_CHOICE and player_idx != self.jester_chooser_idx:
            return False, f"Player {player_idx} not choosing", {}
        
        # Execute action based on current status
        if self.status == GameStatus.IN_PROGRESS:
            return self._execute_play_action(player_idx, action)
        elif self.status == GameStatus.AWAITING_DEFENSE:
            return self._execute_defense_action(player_idx, action)
        elif self.status == GameStatus.AWAITING_JESTER_CHOICE:
            return self._execute_jester_choice(player_idx, action)
        
        return False, "Invalid game state", {}
    
    def _execute_play_action(self, player_idx: int, action: int) -> Tuple[bool, str, Dict]:
        """Execute play action during normal gameplay"""
        hand = self.hands[player_idx]
        
        if action == 0:  # Yield turn
            self._next_turn()
            return True, "Turn yielded", {}
        
        elif 1 <= action <= 5:  # Play single card
            slot = action - 1
            if slot >= len(hand) or hand[slot] == 255:
                return False, "Invalid card slot", {}
            
            card_id = hand[slot]
            success, msg, info = self._play_cards([card_id], player_idx, [slot])
            return success, msg, info
        
        elif 6 <= action <= 15:  # Ace companion
            # Decode ace companion action
            linear_idx = action - 6
            ace_slot = linear_idx // 4
            other_offset = linear_idx % 4
            other_slot = other_offset if other_offset < ace_slot else other_offset + 1
            
            if (ace_slot >= len(hand) or other_slot >= len(hand) or 
                hand[ace_slot] == 255 or hand[other_slot] == 255):
                return False, "Invalid ace companion slots", {}
            
            ace_card = CompactCard(hand[ace_slot])
            if ace_card.rank_idx != 0:
                return False, "First card is not an ace", {}
            
            cards = [hand[ace_slot], hand[other_slot]]
            slots = [ace_slot, other_slot]
            return self._play_cards(cards, player_idx, slots)
        
        elif 16 <= action <= 20:  # Play set
            rank_idx = action - 16 + 1  # Convert to rank index (1-5 for ranks 2-6)
            
            # Find all cards of this rank
            matching_slots = []
            for slot in range(len(hand)):
                if hand[slot] != 255:
                    card = CompactCard(hand[slot])
                    if card.rank_idx == rank_idx:
                        matching_slots.append(slot)
            
            if len(matching_slots) < 2:
                return False, f"Not enough cards of rank {rank_idx+1}", {}
            
            cards = [hand[slot] for slot in matching_slots]
            return self._play_cards(cards, player_idx, matching_slots)
        
        return False, "Invalid action", {}
    
    def _play_cards(self, card_ids: List[int], player_idx: int, slots: List[int]) -> Tuple[bool, str, Dict]:
        """Play cards and resolve effects"""
        total_attack = sum(CompactCard(card_id).attack_power for card_id in card_ids)
        
        # Remove cards from hand
        for slot in slots:
            self.hands[player_idx, slot] = 255
        
        # Check for joker
        has_joker = any(CompactCard(card_id).is_joker for card_id in card_ids)
        if has_joker:
            self.joker_cancels_immunity = True
        
        # Apply attack to enemy
        self.current_enemy_health = max(0, self.current_enemy_health - total_attack)
        
        if self.current_enemy_health <= 0:
            # Enemy defeated
            self._defeat_current_enemy()
            return True, f"Enemy defeated with {total_attack} damage!", {}
        else:
            # Enemy attacks back
            self._enemy_counterattack()
            return True, f"Dealt {total_attack} damage", {}
    
    def _defeat_current_enemy(self):
        """Handle enemy defeat"""
        # Move enemy to hospital
        if self.current_enemy_id != 255:
            self.hospital = np.append(self.hospital, self.current_enemy_id)
        
        # Check win condition
        if len(self.castle_deck) == 0:
            self.status = GameStatus.WON
            return
        
        # Draw next enemy
        if len(self.castle_deck) > 0:
            next_enemy = self.castle_deck[0]
            self.castle_deck = self.castle_deck[1:]
            self._set_current_enemy(next_enemy)
        
        self.joker_cancels_immunity = False  # Reset joker effect
        self._next_turn()
    
    def _enemy_counterattack(self):
        """Handle enemy counterattack"""
        damage = self.current_enemy_attack
        
        # Check immunity (if no joker canceling)
        if not self.joker_cancels_immunity:
            enemy_card = CompactCard(self.current_enemy_id)
            # Enemy is immune to its own suit - damage is doubled
            damage *= 2
        
        # Set up defense phase
        self.damage_to_defend = damage
        self.defend_player_idx = self.current_player  
        self.status = GameStatus.AWAITING_DEFENSE
    
    def _execute_defense_action(self, player_idx: int, action: int) -> Tuple[bool, str, Dict]:
        """Execute defense action"""
        if not (26 <= action <= 29):
            return False, "Invalid defense action", {}
        
        strategy = action - 26
        cards_discarded = self._select_defense_cards(player_idx, strategy)
        
        # Remove cards from hand
        hand = self.hands[player_idx]
        for card_id in cards_discarded:
            for slot in range(len(hand)):
                if hand[slot] == card_id:
                    hand[slot] = 255
                    break
        
        # Calculate defense value
        defense_value = sum(CompactCard(card_id).value for card_id in cards_discarded)
        
        if defense_value >= self.damage_to_defend:
            # Successful defense
            self.status = GameStatus.IN_PROGRESS
            self.damage_to_defend = 0
            self.defend_player_idx = -1
            self._next_turn()
            return True, f"Successfully defended with {defense_value} points", {}
        else:
            # Failed defense - game over
            self.status = GameStatus.LOST
            return False, f"Defense failed: {defense_value} < {self.damage_to_defend}", {}
    
    def _select_defense_cards(self, player_idx: int, strategy: int) -> List[int]:
        """Select cards for defense based on strategy"""
        hand = self.hands[player_idx]
        valid_cards = [card_id for card_id in hand if card_id != 255]
        
        # Sort by value
        valid_cards.sort(key=lambda x: CompactCard(x).value)
        
        if strategy == 0:  # Minimal
            selected = []
            total = 0
            for card_id in valid_cards:
                selected.append(card_id)
                total += CompactCard(card_id).value
                if total >= self.damage_to_defend:
                    break
            return selected
        elif strategy == 1:  # Conservative 
            selected = []
            total = 0
            target = self.damage_to_defend + 2
            for card_id in valid_cards:
                selected.append(card_id)
                total += CompactCard(card_id).value
                if total >= target:
                    break
            return selected
        elif strategy == 2:  # Aggressive
            valid_cards.sort(key=lambda x: CompactCard(x).value, reverse=True)
            selected = []
            total = 0
            for card_id in valid_cards:
                selected.append(card_id)
                total += CompactCard(card_id).value
                if total >= self.damage_to_defend:
                    break
            return selected
        else:  # All
            return valid_cards
    
    def _execute_jester_choice(self, player_idx: int, action: int) -> Tuple[bool, str, Dict]:
        """Execute jester next player choice"""
        if not (0 <= action < self.num_players):
            return False, "Invalid player choice", {}
        
        self.current_player = action
        self.status = GameStatus.IN_PROGRESS
        self.jester_chooser_idx = -1
        
        return True, f"Next player set to {action}", {}
    
    def _next_turn(self):
        """Advance to next player's turn"""
        self.current_player = (self.current_player + 1) % self.num_players
    
    def get_state_dict(self) -> Dict:
        """Get current game state as dictionary"""
        return {
            'status': int(self.status),
            'current_player': self.current_player,
            'current_enemy_id': int(self.current_enemy_id),
            'current_enemy_health': self.current_enemy_health,
            'current_enemy_attack': self.current_enemy_attack, 
            'current_enemy_shield': self.current_enemy_shield,
            'damage_to_defend': self.damage_to_defend,
            'defend_player_idx': self.defend_player_idx,
            'jester_chooser_idx': self.jester_chooser_idx,
            'joker_cancels_immunity': self.joker_cancels_immunity,
            'hands': self.hands.copy(),
            'tavern_deck_size': len(self.tavern_deck),
            'castle_deck_size': len(self.castle_deck),
            'hospital_size': len(self.hospital),
        }


class CompactRegicideEnv(gym.Env):
    """
    Ultra-fast Regicide environment for MARL training.
    
    Optimizations:
    - No I/O operations
    - Vectorized numpy operations
    - Minimal memory allocation
    - Fast action validation
    - Efficient state representation
    """
    
    def __init__(self, num_players: int = 4, render_mode: Optional[str] = None):
        super().__init__()
        
        self.num_players = num_players
        self.render_mode = render_mode
        
        # Compact action space (30 actions max)
        self.action_space = spaces.Discrete(30)
        
        # Efficient observation space (48 features)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(48,), dtype=np.float32
        )
        
        # Game engine
        self.engine = CompactRegicideEngine(num_players)
        self.current_player_idx = 0
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment"""
        super().reset(seed=seed)
        
        if seed is not None:
            self.engine = CompactRegicideEngine(self.num_players, seed)
        else:
            self.engine.reset()
        
        self.current_player_idx = 0
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int):
        """Execute action step"""
        # Execute action
        success, message, action_info = self.engine.step(self.current_player_idx, action)
        
        # Update current player
        self.current_player_idx = self.engine.current_player
        
        # Calculate reward
        reward = self._calculate_reward(success, message, action_info)
        
        # Check termination
        terminated = (self.engine.status == GameStatus.WON or 
                     self.engine.status == GameStatus.LOST)
        
        obs = self._get_observation()
        info = self._get_info()
        info['action_result'] = {'success': success, 'message': message}
        
        return obs, reward, terminated, False, info
    
    def _get_observation(self) -> np.ndarray:
        """Get vectorized observation"""
        obs = np.zeros(48, dtype=np.float32)
        idx = 0
        
        # Player hand (20 features: 5 slots * 4 features each)
        hand = self.engine.hands[self.current_player_idx]
        for slot in range(5):
            if slot < len(hand) and hand[slot] != 255:
                card = CompactCard(hand[slot])
                obs[idx] = card.value / 20.0  # Normalized value
                obs[idx + 1] = float(card.is_joker)
                obs[idx + 2] = card.suit_idx / 4.0 if not card.is_joker else 0.0
                obs[idx + 3] = card.rank_idx / 12.0 if not card.is_joker else 0.0
            idx += 4
        
        # Enemy info (4 features)
        obs[idx] = self.engine.current_enemy_health / 40.0
        obs[idx + 1] = self.engine.current_enemy_attack / 20.0
        obs[idx + 2] = float(self.engine.current_enemy_id != 255)  # Enemy present
        
        if self.engine.current_enemy_id != 255:
            enemy_card = CompactCard(self.engine.current_enemy_id)
            obs[idx + 3] = enemy_card.suit_idx / 4.0
        idx += 4
        
        # Game state (8 features)
        obs[idx] = float(self.engine.status) / 4.0  # Status encoding
        obs[idx + 1] = self.current_player_idx / (self.num_players - 1)  # Current player
        obs[idx + 2] = len(self.engine.tavern_deck) / 50.0  # Tavern size
        obs[idx + 3] = len(self.engine.castle_deck) / 12.0  # Castle size
        obs[idx + 4] = len(self.engine.hospital) / 12.0  # Hospital size
        obs[idx + 5] = float(self.engine.joker_cancels_immunity)  # Joker active
        obs[idx + 6] = self.engine.damage_to_defend / 20.0  # Defense damage
        obs[idx + 7] = float(self.engine.defend_player_idx == self.current_player_idx)  # Is defending
        idx += 8
        
        # Hand summary stats (8 features)
        valid_cards = hand[hand != 255]
        if len(valid_cards) > 0:
            values = [CompactCard(card_id).value for card_id in valid_cards]
            obs[idx] = len(valid_cards) / 8.0  # Hand size
            obs[idx + 1] = np.mean(values) / 20.0  # Average value
            obs[idx + 2] = np.max(values) / 20.0  # Max value
            obs[idx + 3] = np.min(values) / 20.0  # Min value
            
            # Count by suits
            suits = [CompactCard(card_id).suit_idx for card_id in valid_cards if not CompactCard(card_id).is_joker]
            for suit_idx in range(4):
                obs[idx + 4 + suit_idx] = suits.count(suit_idx) / 8.0
        idx += 8
        
        # Action context (8 features - remaining space)
        valid_actions = self.engine.get_valid_actions(self.current_player_idx)
        obs[idx] = len(valid_actions) / 30.0  # Number of valid actions
        
        # Action type availability
        obs[idx + 1] = float(0 in valid_actions)  # Can yield
        obs[idx + 2] = float(any(1 <= a <= 5 for a in valid_actions))  # Can play single
        obs[idx + 3] = float(any(6 <= a <= 15 for a in valid_actions))  # Can play ace combo
        obs[idx + 4] = float(any(16 <= a <= 20 for a in valid_actions))  # Can play set
        obs[idx + 5] = float(any(26 <= a <= 29 for a in valid_actions))  # Can defend
        obs[idx + 6] = float(any(0 <= a <= 3 for a in valid_actions))  # Can choose player
        obs[idx + 7] = 0.0  # Reserved
        
        return obs
    
    def _calculate_reward(self, success: bool, message: str, action_info: Dict) -> float:
        """Calculate reward signal"""
        if self.engine.status == GameStatus.WON:
            return 100.0
        elif self.engine.status == GameStatus.LOST:
            return -100.0
        
        if not success:
            return -1.0  # Invalid action penalty
        
        # Small positive reward for valid actions
        return 0.1
    
    def _get_info(self) -> Dict:
        """Get info dictionary"""
        return {
            'num_players': self.num_players,
            'current_player': self.current_player_idx,
            'game_status': int(self.engine.status),
            'valid_actions': self.engine.get_valid_actions(self.current_player_idx).tolist(),
            'enemy_health': self.engine.current_enemy_health,
            'hand_size': np.sum(self.engine.hands[self.current_player_idx] != 255),
        }
    
    def get_valid_action_mask(self) -> np.ndarray:
        """Get boolean mask for valid actions (for action masking)"""
        mask = np.zeros(self.action_space.n, dtype=bool)
        valid_actions = self.engine.get_valid_actions(self.current_player_idx)
        mask[valid_actions] = True
        return mask
    
    def render(self):
        """Render game state"""
        if self.render_mode == "human":
            state = self.engine.get_state_dict()
            print(f"=== Compact Regicide (Player {self.current_player_idx}) ===")
            print(f"Status: {GameStatus(state['status']).name}")
            print(f"Enemy: {CompactCard(state['current_enemy_id'])} "
                  f"(HP: {state['current_enemy_health']})")
            
            hand = state['hands'][self.current_player_idx]
            hand_cards = [str(CompactCard(card_id)) for card_id in hand if card_id != 255]
            print(f"Hand: {hand_cards}")
            
            valid_actions = self.engine.get_valid_actions(self.current_player_idx)
            print(f"Valid actions: {len(valid_actions)}")
            print("=" * 40)


# Vectorized batch environment for efficient MARL training
class VectorizedRegicideEnv:
    """
    Vectorized environment for training multiple agents simultaneously.
    
    Supports batch reset/step operations for maximum training throughput.
    """
    
    def __init__(self, num_envs: int, num_players: int = 4):
        self.num_envs = num_envs
        self.num_players = num_players
        
        # Create batch of environments
        self.envs = [CompactRegicideEnv(num_players) for _ in range(num_envs)]
        
        # Shared spaces
        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space
    
    def reset(self, seed: Optional[int] = None):
        """Reset all environments"""
        obs_batch = []
        info_batch = []
        
        for i, env in enumerate(self.envs):
            env_seed = seed + i if seed is not None else None
            obs, info = env.reset(seed=env_seed)
            obs_batch.append(obs)
            info_batch.append(info)
        
        return np.array(obs_batch), info_batch
    
    def step(self, actions: np.ndarray):
        """Step all environments"""
        obs_batch = []
        rewards_batch = []
        terminated_batch = []
        truncated_batch = []
        info_batch = []
        
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            obs_batch.append(obs)
            rewards_batch.append(reward)
            terminated_batch.append(terminated)
            truncated_batch.append(truncated)
            info_batch.append(info)
        
        return (np.array(obs_batch), np.array(rewards_batch), 
                np.array(terminated_batch), np.array(truncated_batch), info_batch)
    
    def get_valid_action_masks(self) -> np.ndarray:
        """Get action masks for all environments"""
        masks = []
        for env in self.envs:
            masks.append(env.get_valid_action_mask())
        return np.array(masks)


if __name__ == "__main__":
    # Simple test
    env = CompactRegicideEnv()
    obs, info = env.reset(seed=42)
    
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Valid actions: {info['valid_actions']}")
    
    # Test step
    action = info['valid_actions'][0] if info['valid_actions'] else 0
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Reward: {reward}, Terminated: {terminated}")
    print(f"New valid actions: {info['valid_actions']}")