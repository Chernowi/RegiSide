"""
Regicide Gymnasium Environment

This environment uses a hierarchical action system to reduce the action space
from 5000+ actions to 30 actions, making it suitable for RL training.

The key insight is to separate:
1. Action TYPE (what kind of action: play card, yield, defend, etc.)
2. Action PARAMETERS (which specific cards, players, etc.)

This approach is efficient for RL agents to learn.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any, Set
import random
from enum import Enum

# Import the existing regicide engine
import regicide_engine as engine
from regicide_engine import Card, GameStatusEnum


class ActionType(Enum):
    """Different types of actions in Regicide"""
    PLAY_SINGLE_CARD = 0      # Play one card from hand (+ which card)
    PLAY_ACE_COMPANION = 1    # Play ace + another card (+ which ace, which other)
    PLAY_SET = 2              # Play multiple cards of same rank (+ which rank, how many)
    PLAY_JOKER = 3            # Play a joker alone (+ which joker)
    YIELD_TURN = 4            # Skip turn (no parameters)
    DEFEND = 5                # Discard cards to defend (+ which cards)
    CHOOSE_PLAYER = 6         # Choose next player after jester (+ which player)


class RegicideActionEncoder:
    """
    Action encoder using a hierarchical approach.
    
    Action Space: 
    - Action Type: 7 possible types (0-6)
    - Card Selection: 5 possible card slots (0-4, for cards in hand)  
    - Secondary Selection: 5 possible slots for second card in combinations
    - Player Selection: 4 possible players (0-3)
    - Quantity: 4 possible quantities for sets (1-4 cards)
    
    Uses separate action spaces for different contexts:
    - Normal play: 6 action types + max 5 card choices = ~30 actions
    - Defense: choose card combinations from hand = ~20 actions  
    - Player choice: 4 players = 4 actions
    
    Maximum action space needed: 30 actions
    """
    
    def __init__(self):
        # We'll use a context-dependent action space
        # Normal gameplay: action_type + card_index
        self.MAX_ACTION_SPACE = 30  # Much smaller!
        
        # Action type mappings
        self.ACTION_YIELD = 0
        self.ACTION_PLAY_CARD_START = 1  # Actions 1-5: play cards from hand slots 0-4
        self.ACTION_PLAY_CARD_END = 6
        self.ACTION_ACE_COMPANION_START = 6  # Actions 6-15: ace (slot 0-4) + other card (slot 0-4) 
        self.ACTION_ACE_COMPANION_END = 16
        self.ACTION_PLAY_SET_START = 16  # Actions 16-20: play set of cards of same rank
        self.ACTION_PLAY_SET_END = 21
        self.ACTION_JOKER_START = 21  # Actions 21-25: play joker from slots 0-4
        self.ACTION_JOKER_END = 26
        
        # Defense actions: 26-29 (choose defense strategy)
        self.ACTION_DEFEND_MINIMAL = 26    # Use minimum cards to defend
        self.ACTION_DEFEND_CONSERVATIVE = 27  # Use few extra cards  
        self.ACTION_DEFEND_AGGRESSIVE = 28   # Use many cards to defend
        self.ACTION_DEFEND_ALL = 29          # Use all possible cards
        
        # Player choice actions will use 0-3 when in jester choice mode
        
    def get_action_space_size(self):
        """Get the size of the action space"""
        return self.MAX_ACTION_SPACE
    
    def encode_yield(self) -> int:
        """Encode yielding turn"""
        return self.ACTION_YIELD
    
    def encode_play_card(self, hand_slot: int) -> int:
        """Encode playing a card from a specific hand slot (0-4)"""
        if not 0 <= hand_slot < 5:
            raise ValueError(f"Invalid hand slot: {hand_slot}")
        return self.ACTION_PLAY_CARD_START + hand_slot
    
    def encode_ace_companion(self, ace_slot: int, other_slot: int) -> int:
        """Encode playing ace companion (ace from slot + other card from slot)"""
        if not (0 <= ace_slot < 5 and 0 <= other_slot < 5 and ace_slot != other_slot):
            raise ValueError(f"Invalid ace companion slots: {ace_slot}, {other_slot}")
        # Map to linear index: ace_slot * 5 + other_slot, skipping same slots
        linear_idx = ace_slot * 4 + (other_slot if other_slot < ace_slot else other_slot - 1)
        if linear_idx >= 10:  # We only have 10 slots allocated
            raise ValueError(f"Ace companion index too large: {linear_idx}")
        return self.ACTION_ACE_COMPANION_START + linear_idx
    
    def encode_play_set(self, rank_type: int) -> int:
        """Encode playing a set of cards of the same rank
        rank_type: 0=rank 2, 1=rank 3, 2=rank 4, 3=rank 5"""
        if not 0 <= rank_type < 4:
            raise ValueError(f"Invalid set rank type: {rank_type}")
        return self.ACTION_PLAY_SET_START + rank_type
    
    def encode_joker(self, hand_slot: int) -> int:
        """Encode playing a joker from specific hand slot"""
        if not 0 <= hand_slot < 5:
            raise ValueError(f"Invalid joker slot: {hand_slot}")
        return self.ACTION_JOKER_START + hand_slot
    
    def encode_defense(self, defense_strategy: int) -> int:
        """Encode defense strategy (0=minimal, 1=conservative, 2=aggressive, 3=all)"""
        if not 0 <= defense_strategy < 4:
            raise ValueError(f"Invalid defense strategy: {defense_strategy}")
        return self.ACTION_DEFEND_MINIMAL + defense_strategy
    
    def encode_choose_player(self, player_idx: int) -> int:
        """Encode choosing a player (0-3) - only valid in jester choice context"""
        if not 0 <= player_idx < 4:
            raise ValueError(f"Invalid player index: {player_idx}")
        return player_idx  # Simple 0-3 mapping for player choice context
    
    def decode_action(self, action: int, hand: List[str], game_context: Dict) -> Tuple[str, Dict]:
        """
        Decode action based on current hand and game context
        
        Args:
            action: Integer action to decode
            hand: Current player's hand
            game_context: Current game state info
            
        Returns:
            (action_type, params) tuple
        """
        game_status = game_context.get('status', 'IN_PROGRESS')
        
        # Handle special contexts first
        if game_status == 'AWAITING_JESTER_CHOICE':
            if 0 <= action < 4:
                return "choose_player", {"player_idx": action}
            else:
                raise ValueError(f"Invalid action {action} for jester choice context")
        
        elif game_status == 'AWAITING_DEFENSE':
            if self.ACTION_DEFEND_MINIMAL <= action <= self.ACTION_DEFEND_ALL:
                strategy = action - self.ACTION_DEFEND_MINIMAL
                return "defend", {"strategy": strategy}
            else:
                raise ValueError(f"Invalid action {action} for defense context")
        
        # Normal gameplay actions
        if action == self.ACTION_YIELD:
            return "yield_turn", {}
        
        elif self.ACTION_PLAY_CARD_START <= action < self.ACTION_PLAY_CARD_END:
            hand_slot = action - self.ACTION_PLAY_CARD_START
            if hand_slot >= len(hand):
                raise ValueError(f"Hand slot {hand_slot} out of range (hand size: {len(hand)})")
            card = hand[hand_slot]
            return "play_cards", {"cards": [card]}
        
        elif self.ACTION_ACE_COMPANION_START <= action < self.ACTION_ACE_COMPANION_END:
            linear_idx = action - self.ACTION_ACE_COMPANION_START
            ace_slot = linear_idx // 4
            other_offset = linear_idx % 4
            other_slot = other_offset if other_offset < ace_slot else other_offset + 1
            
            if ace_slot >= len(hand) or other_slot >= len(hand):
                raise ValueError(f"Hand slots out of range: ace={ace_slot}, other={other_slot}")
            
            ace_card = hand[ace_slot]
            other_card = hand[other_slot]
            
            # Verify ace card is actually an ace
            if not ace_card.startswith('A'):
                raise ValueError(f"Card at ace slot {ace_slot} is not an ace: {ace_card}")
            
            return "play_cards", {"cards": [ace_card, other_card]}
        
        elif self.ACTION_PLAY_SET_START <= action < self.ACTION_PLAY_SET_END:
            rank_type = action - self.ACTION_PLAY_SET_START
            target_rank = str(rank_type + 2)  # 0->2, 1->3, 2->4, 3->5
            
            # Find all cards of this rank in hand
            matching_cards = [card for card in hand if card.startswith(target_rank)]
            if len(matching_cards) < 2:
                raise ValueError(f"Not enough cards of rank {target_rank} for set play")
            
            return "play_cards", {"cards": matching_cards}
        
        elif self.ACTION_JOKER_START <= action < self.ACTION_JOKER_END:
            hand_slot = action - self.ACTION_JOKER_START
            if hand_slot >= len(hand):
                raise ValueError(f"Joker slot {hand_slot} out of range")
            
            card = hand[hand_slot]
            if not card.startswith('X'):
                raise ValueError(f"Card at slot {hand_slot} is not a joker: {card}")
            
            return "play_cards", {"cards": [card]}
        
        else:
            raise ValueError(f"Invalid action: {action}")


class RegicideGymEnv(gym.Env):
    """
    Regicide Gymnasium environment with optimized action space.
    
    Key features:
    - Action space reduced from 5000+ to 30 actions maximum
    - Context-dependent action interpretation
    - Action masking for training efficiency
    - Efficient observation space with continuous value + one-hot suit encoding
    """
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        self.render_mode = render_mode
        self.encoder = RegicideActionEncoder()
        
        # Much smaller action space!
        self.action_space = spaces.Discrete(self.encoder.get_action_space_size())
        
        # Simplified observation space
        self._setup_observation_space()
        
        # Game state
        self.room_code = None
        self.players = ["player_0", "player_1", "player_2", "player_3"]
        self.current_player_idx = 0
        self.game_state = None
        
        # Initialize database
        engine.initialize_database()
    
    def _setup_observation_space(self):
        """Setup observation space with value + suit encoding"""
        # New efficient observation:
        # - Hand encoding: 5 slots * 5 features (value + 4 suits) = 25 features  
        # - Game state: status, enemy info, deck info = ~20 features
        # - Context info: damage, player states = ~10 features
        # Total: ~55 features (much more efficient!)
        
        obs_size = (
            5 * 6 +   # Hand: 5 slots * (1 value + 4 suit bits + 1 joker flag) = 30 features
            4 +       # Enemy: health, attack, shield, card_type (4) 
            4 +       # Decks: tavern, castle, hospital sizes, active_effects (4)
            6 +       # Status: game phase one-hot (6)
            4 +       # Current player one-hot (4)
            4 +       # Damage/defense context (4)
            4 +       # Special states: defend_player, chooser, etc (4)
            4         # Enemy suit immunity info (4)
        )  # Total: 60 features
        
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Create new game
        room_code, error = engine.create_room(self.players[0], "Player 0")
        if error:
            raise RuntimeError(f"Failed to create room: {error}")
        
        self.room_code = room_code
        
        # Add other players
        for i in range(1, 4):
            success, error = engine.join_room(room_code, self.players[i], f"Player {i}")
            if not success:
                raise RuntimeError(f"Failed to add player {i}: {error}")
        
        # Start game
        success, error = engine.start_game(room_code, self.players[0])
        if not success:
            raise RuntimeError(f"Failed to start game: {error}")
        
        # Get initial state
        self.game_state, error = engine.get_game_state(room_code)
        if error:
            raise RuntimeError(f"Failed to get initial game state: {error}")
        
        self.current_player_idx = 0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """Execute action step"""
        if self.game_state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        current_player_id = self.players[self.current_player_idx]
        current_hand = self._get_player_hand(current_player_id)
        
        # Decode action with current context
        try:
            action_type, params = self.encoder.decode_action(
                action, current_hand, self.game_state
            )
        except ValueError as e:
            # Invalid action
            observation = self._get_observation()
            return observation, -1.0, False, False, {"error": str(e), "invalid_action": True}
        
        # Execute action
        success, message = self._execute_action(action_type, params, current_player_id)
        
        # Update game state
        self.game_state, error = engine.get_game_state(self.room_code)
        if error:
            terminated = True
            reward = -1.0 if "lost" in message.lower() else 1.0 if "won" in message.lower() else 0.0
        else:
            # Update current player
            if self.game_state.get("current_player_id"):
                try:
                    self.current_player_idx = self.players.index(self.game_state["current_player_id"])
                except ValueError:
                    pass
            
            # Check termination
            status = self.game_state.get("status", "")
            terminated = status in ["WON", "LOST"]
            reward = self._calculate_reward(success, message, terminated, status)
        
        observation = self._get_observation()
        info = self._get_info()
        info["action_result"] = {"success": success, "message": message}
        
        return observation, reward, terminated, False, info
    
    def _execute_action(self, action_type: str, params: Dict, player_id: str) -> Tuple[bool, str]:
        """Execute action - enhanced to handle defense strategies"""
        try:
            if action_type == "play_cards":
                return engine.play_cards(self.room_code, player_id, params["cards"])
            elif action_type == "yield_turn":
                return engine.yield_turn(self.room_code, player_id)
            elif action_type == "defend":
                # Convert strategy to actual card selection
                cards_to_discard = self._select_defense_cards(params["strategy"], player_id)
                return engine.defend_against_attack(self.room_code, player_id, cards_to_discard)
            elif action_type == "choose_player":
                chosen_player_id = self.players[params["player_idx"]]
                return engine.choose_next_player_after_jester(self.room_code, player_id, chosen_player_id)
            else:
                return False, f"Unknown action type: {action_type}"
        except Exception as e:
            return False, f"Action execution failed: {str(e)}"
    
    def _select_defense_cards(self, strategy: int, player_id: str) -> List[str]:
        """Select cards for defense based on strategy"""
        hand = self._get_player_hand(player_id)
        damage = self.game_state.get('damage_to_defend', 0)
        
        # Sort cards by defense value (ascending for minimal strategy)
        def get_defense_value(card_str):
            try:
                card = Card.from_str(card_str)
                return card.get_value()
            except:
                return 0
        
        sorted_hand = sorted(hand, key=get_defense_value)
        
        if strategy == 0:  # Minimal - use exact amount needed
            selected = []
            total_value = 0
            for card in sorted_hand:
                selected.append(card)
                total_value += get_defense_value(card)
                if total_value >= damage:
                    break
            return selected
        
        elif strategy == 1:  # Conservative - 1-2 extra points
            selected = []
            total_value = 0
            target = damage + 2
            for card in sorted_hand:
                selected.append(card)
                total_value += get_defense_value(card)
                if total_value >= target:
                    break
            return selected
        
        elif strategy == 2:  # Aggressive - use higher value cards
            sorted_hand_desc = sorted(hand, key=get_defense_value, reverse=True)
            selected = []
            total_value = 0
            for card in sorted_hand_desc:
                selected.append(card)
                total_value += get_defense_value(card)
                if total_value >= damage:
                    break
            return selected
        
        else:  # strategy == 3: All possible cards
            return hand.copy()
    
    def _calculate_reward(self, success: bool, message: str, terminated: bool, status: str) -> float:
        """Calculate reward - same as before"""
        if terminated:
            if status == "WON":
                return 100.0
            elif status == "LOST":
                return -100.0
        
        if not success:
            return -1.0
        
        return 0.1
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation vector"""
        if self.game_state is None:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        obs = []
        
        # Current player's hand (5 slots * 6 features each = 30 total)
        current_hand = self._get_player_hand(self.players[self.current_player_idx])
        hand_encoding = self._encode_hand_compact(current_hand)
        obs.extend(hand_encoding)
        
        # Enemy info (4 features)
        enemy_health = self.game_state.get("current_enemy_health", 0) / 40.0
        enemy_attack = self.game_state.get("current_enemy_attack", 0) / 20.0
        enemy_shield = self.game_state.get("current_enemy_shield", 0) / 20.0
        enemy_type = self._encode_enemy_type()  # 0=Jack, 0.5=Queen, 1.0=King
        obs.extend([enemy_health, enemy_attack, enemy_shield, enemy_type])
        
        # Deck info (4 features)  
        tavern_size = self.game_state.get("tavern_deck_size", 0) / 50.0
        castle_size = self.game_state.get("castle_deck_size", 0) / 12.0
        hospital_size = self.game_state.get("hospital_size", 0) / 20.0
        joker_active = float(self.game_state.get("active_joker_cancels_immunity", False))
        obs.extend([tavern_size, castle_size, hospital_size, joker_active])
        
        # Game status (6 one-hot features)
        status_encoding = self._encode_status()
        obs.extend(status_encoding)
        
        # Current player (4 one-hot features)
        player_encoding = [0.0] * 4
        player_encoding[self.current_player_idx] = 1.0
        obs.extend(player_encoding)
        
        # Context info (4 features)
        damage = self.game_state.get("damage_to_defend", 0) / 20.0
        obs.extend([damage, 0.0, 0.0, 0.0])  # Placeholder for other context
        
        # Special states (4 features) 
        defend_active = float(self.game_state.get("status") == "AWAITING_DEFENSE")
        jester_active = float(self.game_state.get("status") == "AWAITING_JESTER_CHOICE")
        obs.extend([defend_active, jester_active, 0.0, 0.0])
        
        # Enemy suit immunity (4 features: H, D, S, C immunity flags)
        enemy_immunity = self._encode_enemy_suit_immunity()
        obs.extend(enemy_immunity)
        
        return np.array(obs, dtype=np.float32)
    
    def _encode_hand_compact(self, hand: List[str]) -> List[float]:
        """Encode hand using continuous value + one-hot suit + joker flag encoding"""
        # Each card: [value, H, D, S, C, joker_flag] = 6 features per slot
        # 5 slots in hand = 30 features total
        
        # Rank values: A=0, 2=1/12, 3=2/12, ..., K=1.0
        rank_values = {
            'A': 0.0, '2': 1/12, '3': 2/12, '4': 3/12, '5': 4/12, '6': 5/12,
            '7': 6/12, '8': 7/12, '9': 8/12, '10': 9/12, 'J': 10/12, 'Q': 11/12, 'K': 1.0
        }
        
        # Suit one-hot encoding
        suit_encoding = {'H': [1,0,0,0], 'D': [0,1,0,0], 'S': [0,0,1,0], 'C': [0,0,0,1]}
        
        encoding = []
        
        for slot_idx in range(5):  # Always encode 5 slots
            if slot_idx < len(hand) and hand[slot_idx]:
                card = hand[slot_idx]
                
                if card.startswith('X'):  # Joker
                    # Special encoding: mid-value + no suit + joker flag
                    encoding.extend([0.5, 0, 0, 0, 0, 1])
                else:
                    # Regular card: value + suit + no joker flag
                    rank = card[:-1]  # All but last character
                    suit = card[-1]   # Last character
                    
                    value = rank_values.get(rank, 0.0)
                    suit_vec = suit_encoding.get(suit, [0,0,0,0])
                    
                    encoding.extend([value] + suit_vec + [0])
            else:
                # Empty slot: all zeros
                encoding.extend([0.0, 0, 0, 0, 0, 0])
        
        return encoding
    
    def _encode_enemy_suit_immunity(self) -> List[float]:
        """Encode which suits the current enemy is immune to"""
        immunity = [0.0, 0.0, 0.0, 0.0]  # [H, D, S, C]
        
        enemy = self.game_state.get("current_enemy", "")
        joker_cancels = self.game_state.get("active_joker_cancels_immunity", False)
        
        if not joker_cancels and enemy:
            # Extract enemy suit (last character of enemy card)
            if len(enemy) > 0:
                enemy_suit = enemy[-1]
                suit_index = {'H': 0, 'D': 1, 'S': 2, 'C': 3}.get(enemy_suit, -1)
                if suit_index >= 0:
                    immunity[suit_index] = 1.0
        
        return immunity
    
    def _encode_enemy_type(self) -> float:
        """Encode enemy type as a single value"""
        enemy = self.game_state.get("current_enemy", "")
        if 'J' in enemy:
            return 0.0  # Jack
        elif 'Q' in enemy:
            return 0.5  # Queen
        elif 'K' in enemy:
            return 1.0  # King
        else:
            return 0.0  # Default
    
    def _encode_status(self) -> List[float]:
        """Encode game status as one-hot"""
        status_map = {
            "WAITING_FOR_PLAYERS": 0, "IN_PROGRESS": 1, "AWAITING_DEFENSE": 2,
            "AWAITING_JESTER_CHOICE": 3, "WON": 4, "LOST": 5
        }
        encoding = [0.0] * 6
        current_status = self.game_state.get("status", "IN_PROGRESS")
        if current_status in status_map:
            encoding[status_map[current_status]] = 1.0
        return encoding
    
    def _get_player_hand(self, player_id: str) -> List[str]:
        """Get player hand - same as before"""
        if self.game_state is None:
            return []
        
        for player_info in self.game_state.get("players", []):
            if player_info["id"] == player_id:
                return player_info.get("hand", [])
        
        return []
    
    def _get_info(self) -> Dict:
        """Get environment info dictionary"""
        info = {
            "room_code": self.room_code,
            "current_player": self.current_player_idx,
            "game_status": self.game_state.get("status") if self.game_state else None,
            "valid_actions": self.get_valid_actions(),
            "action_space_size": self.action_space.n,
            "observation_size": self.observation_space.shape[0]
        }
        return info
    
    def get_valid_actions(self) -> List[int]:
        """Get valid actions for current state"""
        if self.game_state is None:
            return []
        
        valid_actions = []
        current_player_hand = self._get_player_hand(self.players[self.current_player_idx])
        status = self.game_state.get("status", "")
        
        if status == "IN_PROGRESS":
            # Always can yield
            valid_actions.append(self.encoder.encode_yield())
            
            # Can play cards from hand slots
            for i, card in enumerate(current_player_hand[:5]):
                if card:  # If slot is not empty
                    if card.startswith('X'):  # Joker
                        valid_actions.append(self.encoder.encode_joker(i))
                    else:  # Regular card
                        valid_actions.append(self.encoder.encode_play_card(i))
            
            # Ace companions
            ace_slots = [i for i, card in enumerate(current_player_hand[:5]) if card.startswith('A')]
            other_slots = [i for i, card in enumerate(current_player_hand[:5]) if card and not card.startswith('A') and not card.startswith('X')]
            
            for ace_slot in ace_slots:
                for other_slot in other_slots:
                    try:
                        valid_actions.append(self.encoder.encode_ace_companion(ace_slot, other_slot))
                    except ValueError:
                        pass
            
            # Set plays
            for rank_type in range(4):  # ranks 2,3,4,5
                target_rank = str(rank_type + 2)
                matching_count = sum(1 for card in current_player_hand if card.startswith(target_rank))
                if matching_count >= 2:
                    valid_actions.append(self.encoder.encode_play_set(rank_type))
                    
        elif status == "AWAITING_DEFENSE":
            # Defense strategies
            for strategy in range(4):
                valid_actions.append(self.encoder.encode_defense(strategy))
                
        elif status == "AWAITING_JESTER_CHOICE":
            # Player choices
            for player_idx in range(4):
                valid_actions.append(self.encoder.encode_choose_player(player_idx))
        
        return valid_actions
    
    def render(self):
        """Render current game state"""
        if self.render_mode == "human":
            if self.game_state:
                print(f"=== Regicide Game State ===")
                print(f"Room: {self.room_code}")
                print(f"Status: {self.game_state.get('status')}")
                print(f"Current Player: {self.current_player_idx} ({self.players[self.current_player_idx]})")
                
                enemy = self.game_state.get('current_enemy')
                if enemy:
                    print(f"Enemy: {enemy} (HP: {self.game_state.get('current_enemy_health')}, "
                          f"ATK: {self.game_state.get('current_enemy_attack')})")
                
                current_hand = self._get_player_hand(self.players[self.current_player_idx])
                print(f"Your hand: {current_hand}")
                
                valid_actions = self.get_valid_actions()
                print(f"Valid actions: {len(valid_actions)} available")
                print(f"Action space size: {self.action_space.n}")
                print(f"Observation size: {self.observation_space.shape[0]}")
                print("=" * 30)
    
    def close(self):
        """Clean up"""
        pass


