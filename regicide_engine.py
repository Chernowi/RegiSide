import random
import uuid
from enum import Enum, auto
from typing import List, Dict, Optional, Tuple, Any

# --- Constants ---
SUITS = ["H", "D", "S", "C"]  # Hearts, Diamonds, Spades, Clubs
RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
JOKER_RANK = "X" # Using X for Joker rank to distinguish

MAX_PLAYERS = 4
MIN_PLAYERS = 1

ROYAL_HEALTH = {"J": 20, "Q": 30, "K": 40}
ROYAL_ATTACK = {"J": 10, "Q": 15, "K": 20}

# Hand sizes per player count
HAND_SIZES = {
    1: 8,
    2: 7,
    3: 6,
    4: 5,
}

# Jokers in Tavern Deck per player count
JOKERS_IN_DECK = {
    1: 0, # Solo jokers are special
    2: 0,
    3: 1,
    4: 2,
}

class Card:
    def __init__(self, suit: Optional[str], rank: str):
        self.suit = suit # None for Joker
        self.rank = rank

    def __repr__(self):
        return f"{self.rank}{self.suit if self.suit else ''}"

    def get_value(self, for_attack=False) -> int:
        if self.rank == JOKER_RANK:
            return 0 # Jokers have 0 value for absorbing damage, no attack value on their own
        if self.rank == "A":
            return 1
        if self.rank.isdigit():
            return int(self.rank)
        if self.rank == "J":
            return 10 if not for_attack else ROYAL_ATTACK["J"] # Attack value is different if it's the enemy
        if self.rank == "Q":
            return 15 if not for_attack else ROYAL_ATTACK["Q"]
        if self.rank == "K":
            return 20 if not for_attack else ROYAL_ATTACK["K"]
        return 0

    def get_attack_power(self) -> int: # Attack power when played from hand
        if self.rank == "A": return 1
        if self.rank.isdigit(): return int(self.rank)
        if self.rank == "J": return 10 # J, Q, K from hand have fixed attack values
        if self.rank == "Q": return 15
        if self.rank == "K": return 20
        return 0 # Joker doesn't attack directly

class GameStatus(Enum):
    WAITING_FOR_PLAYERS = auto()
    IN_PROGRESS = auto()
    WON = auto()
    LOST = auto()

class Player:
    def __init__(self, player_id: str, player_name: str):
        self.id: str = player_id
        self.name: str = player_name
        self.hand: List[Card] = []

    def __repr__(self):
        return f"Player({self.name}, Hand: {self.hand})"

class GameState:
    def __init__(self, room_code: str, created_by: str):
        self.room_code: str = room_code
        self.players: Dict[str, Player] = {} # player_id -> Player object
        self.player_order: List[str] = [] # List of player_ids in turn order
        self.status: GameStatus = GameStatus.WAITING_FOR_PLAYERS
        self.created_by: str = created_by # player_id of creator

        self.tavern_deck: List[Card] = []
        self.castle_deck: List[Card] = []
        self.hospital: List[Card] = [] # Discard pile

        self.current_enemy: Optional[Card] = None
        self.current_enemy_health: int = 0
        self.current_enemy_base_attack: int = 0 # Base attack of the current enemy
        self.current_enemy_shield: int = 0 # Damage reduction from Spades

        self.current_player_idx: int = 0
        self.active_joker_cancels_immunity: bool = False
        self.consecutive_yields: int = 0

        # Solo mode specific
        self.solo_jokers_available: int = 2 if len(self.players) == 1 and MIN_PLAYERS == 1 else 0

    def get_player(self, player_id: str) -> Optional[Player]:
        return self.players.get(player_id)

    def get_current_player(self) -> Optional[Player]:
        if not self.player_order:
            return None
        return self.players.get(self.player_order[self.current_player_idx])

    def to_dict(self, perspective_player_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the game state.
        Hides other players' hands if perspective_player_id is provided.
        """
        players_info = []
        for pid, player in self.players.items():
            if perspective_player_id == pid:
                players_info.append({"id": pid, "name": player.name, "hand": [str(c) for c in player.hand], "hand_size": len(player.hand)})
            else:
                players_info.append({"id": pid, "name": player.name, "hand_size": len(player.hand)})


        return {
            "room_code": self.room_code,
            "status": self.status.name,
            "players": players_info,
            "player_order": self.player_order,
            "tavern_deck_size": len(self.tavern_deck),
            "castle_deck_size": len(self.castle_deck),
            "hospital_size": len(self.hospital),
            "current_enemy": str(self.current_enemy) if self.current_enemy else None,
            "current_enemy_health": self.current_enemy_health,
            "current_enemy_attack": max(0, self.current_enemy_base_attack - self.current_enemy_shield),
            "current_enemy_shield": self.current_enemy_shield,
            "current_player_id": self.player_order[self.current_player_idx] if self.player_order else None,
            "active_joker_cancels_immunity": self.active_joker_cancels_immunity,
            "consecutive_yields": self.consecutive_yields,
            "solo_jokers_available": self.solo_jokers_available if len(self.players)==1 else None,
        }

# --- Global Game Storage (In-memory, replace with DB for persistence) ---
games: Dict[str, GameState] = {}

# --- Helper Functions ---
def _create_deck() -> List[Card]:
    deck = [Card(suit, rank) for suit in SUITS for rank in RANKS if rank not in ["J", "Q", "K"]]
    return deck

def _create_castle_deck() -> List[Card]:
    jacks = [Card(s, "J") for s in SUITS]
    queens = [Card(s, "Q") for s in SUITS]
    kings = [Card(s, "K") for s in SUITS]
    random.shuffle(jacks)
    random.shuffle(queens)
    random.shuffle(kings)
    return jacks + queens + kings # Jacks on top, then Queens, then Kings

def _get_card_from_str(card_str: str) -> Card:
    """Converts a string like 'H10' or 'SJ' or 'X' (for Joker) to a Card object."""
    if card_str.upper() == JOKER_RANK:
        return Card(None, JOKER_RANK)
    rank = card_str[:-1]
    suit = card_str[-1:].upper()
    if suit not in SUITS or rank not in RANKS:
        raise ValueError(f"Invalid card string: {card_str}")
    return Card(suit, rank)

def _deal_cards(game: GameState):
    num_players = len(game.players)
    if not num_players: return

    max_hand = HAND_SIZES.get(num_players, 5) # Default to 5 if somehow out of range
    for _ in range(max_hand):
        for player_id in game.player_order:
            player = game.players[player_id]
            if len(player.hand) < max_hand and game.tavern_deck:
                player.hand.append(game.tavern_deck.pop())

# --- API Functions ---

def create_room(player_id: str, player_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Creates a new game room.
    Returns (room_code, error_message).
    """
    room_code = str(uuid.uuid4())[:6].upper() # Simple room code
    while room_code in games:
        room_code = str(uuid.uuid4())[:6].upper()

    game = GameState(room_code, player_id)
    games[room_code] = game
    # Automatically join the creator
    success, error = join_room(room_code, player_id, player_name)
    if not success:
        del games[room_code] # Clean up if join fails (shouldn't happen here)
        return None, error
    return room_code, None

def join_room(room_code: str, player_id: str, player_name: str) -> Tuple[bool, Optional[str]]:
    """
    Adds a player to an existing game room.
    Returns (success_boolean, error_message).
    """
    game = games.get(room_code)
    if not game:
        return False, "Room not found."
    if game.status != GameStatus.WAITING_FOR_PLAYERS:
        return False, "Game has already started."
    if len(game.players) >= MAX_PLAYERS:
        return False, "Room is full."
    if player_id in game.players:
        return False, "Player already in room." # Or allow reconnect logic

    game.players[player_id] = Player(player_id, player_name)
    game.player_order.append(player_id) # Add to order
    return True, None

def get_game_state(room_code: str, perspective_player_id: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Retrieves the current state of the game.
    Returns (game_state_dict, error_message).
    """
    game = games.get(room_code)
    if not game:
        return None, "Room not found."
    return game.to_dict(perspective_player_id), None


def start_game(room_code: str, requesting_player_id: str) -> Tuple[bool, Optional[str]]:
    """
    Starts the game if conditions are met.
    Returns (success_boolean, error_message).
    """
    game = games.get(room_code)
    if not game:
        return False, "Room not found."
    if game.created_by != requesting_player_id:
        return False, "Only the room creator can start the game."
    if game.status != GameStatus.WAITING_FOR_PLAYERS:
        return False, "Game already started or finished."
    if len(game.players) < MIN_PLAYERS:
        return False, f"Not enough players. Need at least {MIN_PLAYERS}."

    # Initialize Decks
    game.castle_deck = _create_castle_deck()
    game.tavern_deck = _create_deck()

    num_jokers_to_add = JOKERS_IN_DECK.get(len(game.players), 0)
    for _ in range(num_jokers_to_add):
        game.tavern_deck.append(Card(None, JOKER_RANK))

    random.shuffle(game.tavern_deck)

    # Solo mode Jokers
    if len(game.players) == 1:
        game.solo_jokers_available = 2 # These are separate from deck jokers

    # Deal Cards
    _deal_cards(game)

    # Set up first enemy
    if not game.castle_deck:
        game.status = GameStatus.WON # Should not happen with a full castle deck
        return False, "Error: Castle deck empty at start."

    game.current_enemy = game.castle_deck.pop(0) # Draw from the top (Jacks first)
    game.current_enemy_health = ROYAL_HEALTH[game.current_enemy.rank]
    game.current_enemy_base_attack = ROYAL_ATTACK[game.current_enemy.rank]
    game.current_enemy_shield = 0
    game.active_joker_cancels_immunity = False

    game.status = GameStatus.IN_PROGRESS
    game.current_player_idx = 0 # First player in order starts
    game.consecutive_yields = 0

    return True, None

def _advance_turn(game: GameState):
    game.current_player_idx = (game.current_player_idx + 1) % len(game.players)

def _check_game_over(game: GameState) -> bool:
    if game.status == GameStatus.WON or game.status == GameStatus.LOST:
        return True
    return False

def _handle_enemy_defeat(game: GameState, overkill: bool):
    defeated_enemy = game.current_enemy
    game.current_enemy = None # Clear current enemy
    game.active_joker_cancels_immunity = False # Reset joker effect

    if overkill:
        game.hospital.append(defeated_enemy)
    else: # Exact kill
        game.tavern_deck.insert(0, defeated_enemy) # Place on top of tavern deck

    if not game.castle_deck: # No more enemies
        game.status = GameStatus.WON
        return

    # Set up next enemy
    game.current_enemy = game.castle_deck.pop(0)
    game.current_enemy_health = ROYAL_HEALTH[game.current_enemy.rank]
    game.current_enemy_base_attack = ROYAL_ATTACK[game.current_enemy.rank]
    game.current_enemy_shield = 0
    # Current player takes another turn, Royal does not attack this interim

def play_cards(room_code: str, player_id: str, card_strs_played: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Player plays one or more cards.
    card_strs_played: List of strings like ["H5", "C5"] or ["SA"] or ["X"]
    """
    game = games.get(room_code)
    if not game: return False, "Room not found."
    if _check_game_over(game): return False, f"Game is over: {game.status.name}"
    if game.status != GameStatus.IN_PROGRESS: return False, "Game not started."

    player = game.get_player(player_id)
    current_player = game.get_current_player()
    if not player or player != current_player:
        return False, "Not your turn or player not found."
    if not game.current_enemy:
        return False, "No active enemy." # Should not happen in IN_PROGRESS

    played_cards: List[Card] = []
    try:
        for cs in card_strs_played:
            card = _get_card_from_str(cs)
            # Check if player has the card
            found_in_hand = False
            for hand_card_idx, hand_card in enumerate(player.hand):
                if hand_card.rank == card.rank and hand_card.suit == card.suit:
                    played_cards.append(player.hand.pop(hand_card_idx))
                    found_in_hand = True
                    break
            if not found_in_hand:
                # Rollback cards removed from hand if one is missing
                for p_card in played_cards: player.hand.append(p_card)
                return False, f"Card {cs} not in hand."
    except ValueError as e:
        return False, str(e)

    if not played_cards:
        return False, "No cards selected to play."

    # --- Validate Play ---
    is_joker_play = any(c.rank == JOKER_RANK for c in played_cards)
    is_animal_companion = False
    is_set_play = False
    total_attack_value = 0

    if is_joker_play:
        if len(played_cards) > 1:
            for p_card in played_cards: player.hand.append(p_card) # Return cards to hand
            return False, "Joker must be played alone."
        joker = played_cards[0]
        game.active_joker_cancels_immunity = True
        game.hospital.append(joker) # Joker goes to hospital
        # Player who played Joker chooses next player (simplification: next player in order)
        # Or you could add another param: next_player_id
        # For now, turn advances, Royal does NOT attack.
        _advance_turn(game)
        game.consecutive_yields = 0
        return True, "Joker played. Immunity cancelled for this enemy. Royal does not attack."

    # Not a Joker play
    if len(played_cards) == 1:
        total_attack_value = played_cards[0].get_attack_power()
    elif len(played_cards) == 2 and any(c.rank == "A" for c in played_cards): # Animal Companion
        ace = next(c for c in played_cards if c.rank == "A")
        other_card = next(c for c in played_cards if c.rank != "A")
        if other_card.rank == JOKER_RANK: # Ace cannot be played with Joker as companion
            for p_card in played_cards: player.hand.append(p_card)
            return False, "Ace cannot be companioned with a Joker."
        total_attack_value = other_card.get_attack_power() + ace.get_attack_power() # Ace adds 1 effectively
        is_animal_companion = True
    else: # Set play
        is_set_play = True
        first_rank = played_cards[0].rank
        if not all(c.rank == first_rank for c in played_cards):
            for p_card in played_cards: player.hand.append(p_card)
            return False, "Set play cards must be of the same rank."
        if not first_rank.isdigit() or not (2 <= int(first_rank) <= 5):
            for p_card in played_cards: player.hand.append(p_card)
            return False, "Set play only allowed for ranks 2, 3, 4, 5."
        total_attack_value = sum(c.get_attack_power() for c in played_cards)
        if total_attack_value > 10:
            for p_card in played_cards: player.hand.append(p_card)
            return False, "Set play attack sum cannot exceed 10."

    game.consecutive_yields = 0 # Reset yields on valid play

    # --- Activate Suit Powers ---
    actual_damage = total_attack_value
    enemy_immune_suit = game.current_enemy.suit
    can_activate_power = lambda suit: not (suit == enemy_immune_suit and not game.active_joker_cancels_immunity)

    for card in played_cards:
        if card.suit == "H" and can_activate_power("H"): # Hearts (Heal)
            # Shuffle hospital, draw X, put on bottom of tavern
            random.shuffle(game.hospital)
            num_to_heal = min(total_attack_value, len(game.hospital))
            healed_cards = [game.hospital.pop(0) for _ in range(num_to_heal)]
            game.tavern_deck.extend(healed_cards) # Add to bottom

        elif card.suit == "D" and can_activate_power("D"): # Diamonds (Draw)
            drawn_count = 0
            # Start with current player, then clockwise
            # Need to iterate through players in order from current
            start_idx = game.current_player_idx
            for i in range(len(game.players)):
                if drawn_count >= total_attack_value: break
                p_idx = (start_idx + i) % len(game.players)
                p_to_draw = game.players[game.player_order[p_idx]]
                max_h = HAND_SIZES[len(game.players)]
                while len(p_to_draw.hand) < max_h and game.tavern_deck and drawn_count < total_attack_value:
                    p_to_draw.hand.append(game.tavern_deck.pop())
                    drawn_count += 1

        elif card.suit == "S" and can_activate_power("S"): # Spades (Shield)
            game.current_enemy_shield += total_attack_value # Cumulative

        elif card.suit == "C" and can_activate_power("C"): # Clubs (Double Damage)
            actual_damage *= 2 # Only applies once even if multiple clubs

    # --- Deal Damage to Royal ---
    game.current_enemy_health -= actual_damage
    # Played cards go to hospital
    for pc in played_cards: game.hospital.append(pc)


    # --- Check if Royal is Defeated ---
    if game.current_enemy_health <= 0:
        overkill = game.current_enemy_health < 0
        _handle_enemy_defeat(game, overkill)
        if game.status == GameStatus.WON:
            return True, "Last King defeated! Players win!"
        # Current player continues turn against new enemy (Royal does not attack)
        return True, f"Enemy defeated! New enemy: {game.current_enemy}. Your turn again."
    else:
        # --- Royal Attacks ---
        royal_attack_power = max(0, game.current_enemy_base_attack - game.current_enemy_shield)
        player_defense_value = 0
        cards_to_discard_indices = [] # Store indices to remove later

        # Player must discard cards >= royal_attack_power
        # For this API, the frontend would send which cards to discard.
        # Here, we'll assume an auto-discard for simplicity or expect another call.
        # For a real API, you'd need a separate `discard_for_damage(room_code, player_id, card_strs_to_discard)`
        # This is a simplification:
        temp_hand_for_discard = sorted(player.hand, key=lambda c: c.get_value(), reverse=True)
        discarded_for_damage: List[Card] = []

        for card_in_hand in temp_hand_for_discard:
            if player_defense_value >= royal_attack_power:
                break
            player_defense_value += card_in_hand.get_value()
            discarded_for_damage.append(card_in_hand)

        if player_defense_value < royal_attack_power:
            game.status = GameStatus.LOST
            # Put cards back if logic failed (though here it's a loss anyway)
            return False, f"Player {player.name} cannot withstand attack ({player_defense_value}/{royal_attack_power}). Game Over."
        else:
            # Successfully discarded: remove from actual hand and add to hospital
            for d_card in discarded_for_damage:
                for i, h_card in enumerate(player.hand):
                    if d_card.suit == h_card.suit and d_card.rank == h_card.rank: # Find and remove one instance
                        game.hospital.append(player.hand.pop(i))
                        break
            _advance_turn(game)
            return True, f"Damage dealt. Enemy has {game.current_enemy_health} HP. Royal attacked for {royal_attack_power}. Next player's turn."

def yield_turn(room_code: str, player_id: str) -> Tuple[bool, Optional[str]]:
    game = games.get(room_code)
    if not game: return False, "Room not found."
    if _check_game_over(game): return False, f"Game is over: {game.status.name}"
    if game.status != GameStatus.IN_PROGRESS: return False, "Game not started."

    player = game.get_player(player_id)
    current_player = game.get_current_player()
    if not player or player != current_player:
        return False, "Not your turn or player not found."
    if not game.current_enemy:
        return False, "No active enemy."

    game.consecutive_yields += 1
    if game.consecutive_yields >= len(game.players):
        game.status = GameStatus.LOST
        return False, "All players yielded consecutively. Game Over."

    # Royal Attacks (same logic as after playing cards)
    royal_attack_power = max(0, game.current_enemy_base_attack - game.current_enemy_shield)
    player_defense_value = 0
    # Simplification: auto-discard highest value cards
    temp_hand_for_discard = sorted(player.hand, key=lambda c: c.get_value(), reverse=True)
    discarded_for_damage: List[Card] = []

    for card_in_hand in temp_hand_for_discard:
        if player_defense_value >= royal_attack_power:
            break
        player_defense_value += card_in_hand.get_value()
        discarded_for_damage.append(card_in_hand)

    if player_defense_value < royal_attack_power:
        game.status = GameStatus.LOST
        return False, f"Player {player.name} cannot withstand attack ({player_defense_value}/{royal_attack_power}) after yielding. Game Over."
    else:
        for d_card in discarded_for_damage:
            for i, h_card in enumerate(player.hand):
                if d_card.suit == h_card.suit and d_card.rank == h_card.rank:
                    game.hospital.append(player.hand.pop(i))
                    break
        _advance_turn(game)
        return True, f"Yielded. Royal attacked for {royal_attack_power}. Next player's turn."

def use_solo_joker_power(room_code: str, player_id: str) -> Tuple[bool, Optional[str]]:
    game = games.get(room_code)
    if not game: return False, "Room not found."
    if _check_game_over(game): return False, f"Game is over: {game.status.name}"
    if game.status != GameStatus.IN_PROGRESS: return False, "Game not started."
    if len(game.players) != 1: return False, "Solo Joker power only for solo games."

    player = game.get_player(player_id)
    current_player = game.get_current_player()
    if not player or player != current_player: return False, "Not your turn or player not found."

    if game.solo_jokers_available <= 0:
        return False, "No solo Jokers available."

    game.solo_jokers_available -= 1
    # Discard hand
    game.hospital.extend(player.hand)
    player.hand = []
    # Draw back up to 8
    max_hand = HAND_SIZES[1]
    while len(player.hand) < max_hand and game.tavern_deck:
        player.hand.append(game.tavern_deck.pop())

    return True, f"Solo Joker used. Hand refreshed. {game.solo_jokers_available} solo Jokers remaining."

# --- Example Usage (Conceptual - This would be called by your interface) ---
if __name__ == '__main__':
    # --- Room Creation and Joining ---
    p1_id, p1_name = "player1", "Alice"
    p2_id, p2_name = "player2", "Bob"

    room_code, error = create_room(p1_id, p1_name)
    if error:
        print(f"Error creating room: {error}")
        exit()
    print(f"Room '{room_code}' created by {p1_name}.")

    success, error = join_room(room_code, p2_id, p2_name)
    if error:
        print(f"Error {p2_name} joining room: {error}")
        exit()
    print(f"{p2_name} joined room '{room_code}'.")

    state, error = get_game_state(room_code, p1_id)
    # print("Initial State:", state)

    # --- Start Game ---
    success, error = start_game(room_code, p1_id) # Only creator can start
    if error:
        print(f"Error starting game: {error}")
        exit()
    print("Game started!")

    state, error = get_game_state(room_code, p1_id)
    # print("State after start for Alice:", state)
    # current_player_id = state['current_player_id']
    # current_enemy_str = state['current_enemy']
    # print(f"Current player: {current_player_id}, Current Enemy: {current_enemy_str}")


    # --- Gameplay Loop (Simplified - a real interface would drive this) ---
    game_over = False
    turn_count = 0
    while not game_over and turn_count < 50: # Safety break
        turn_count += 1
        current_state, _ = get_game_state(room_code)
        if not current_state: break
        if current_state['status'] not in [GameStatus.IN_PROGRESS.name]:
            print(f"Game ended: {current_state['status']}")
            game_over = True
            break

        cp_id = current_state['current_player_id']
        cp_obj = games[room_code].players[cp_id]
        print(f"\n--- Turn {turn_count} ---")
        print(f"Current Player: {cp_obj.name} (Hand: {[str(c) for c in cp_obj.hand]})")
        print(f"Enemy: {current_state['current_enemy']} (HP: {current_state['current_enemy_health']}, ATK: {current_state['current_enemy_attack']})")
        print(f"Tavern: {current_state['tavern_deck_size']}, Hospital: {current_state['hospital_size']}")

        # AI: Simplistic play - play first valid card or yield if hand empty
        if not cp_obj.hand:
            print(f"{cp_obj.name} has no cards, yielding.")
            success, message = yield_turn(room_code, cp_id)
        else:
            # Try to play the first card in hand
            card_to_play_str = str(cp_obj.hand[0]) # This is a simplification; need to find it by value
            print(f"{cp_obj.name} plays {card_to_play_str}")
            success, message = play_cards(room_code, cp_id, [card_to_play_str])

        print(message)
        if not success and ("Game Over" in message or "win" in message):
            game_over = True

    final_state, _ = get_game_state(room_code)
    print("\n--- Final Game State ---")
    print(final_state)