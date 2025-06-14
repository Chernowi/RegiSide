import os
import random
import uuid
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any

from sqlalchemy import create_engine, Column, String, Integer, Boolean, JSON, ForeignKey, DateTime, Enum as SQLAEnum, inspect
from sqlalchemy.orm import sessionmaker, Session, relationship, declarative_base
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB # Use JSONB for PostgreSQL

# --- Constants ---
SUITS = ["H", "D", "S", "C"]  # Hearts, Diamonds, Spades, Clubs
RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
JOKER_RANK = "X"

MAX_PLAYERS_CONST = 4 # Renamed to avoid conflict
MIN_PLAYERS_CONST = 1

ROYAL_HEALTH_CONST = {"J": 20, "Q": 30, "K": 40}
ROYAL_ATTACK_CONST = {"J": 10, "Q": 15, "K": 20}

HAND_SIZES_CONST = {1: 8, 2: 7, 3: 6, 4: 5}
JOKERS_IN_DECK_CONST = {1: 0, 2: 0, 3: 1, 4: 2}

class GameStatusEnum(Enum):
    WAITING_FOR_PLAYERS = "WAITING_FOR_PLAYERS"
    IN_PROGRESS = "IN_PROGRESS"
    AWAITING_DEFENSE = "AWAITING_DEFENSE"
    AWAITING_JESTER_CHOICE = "AWAITING_JESTER_CHOICE"
    WON = "WON"
    LOST = "LOST"

# --- Card Class & Utilities (essential for game logic) ---
class Card:
    def __init__(self, suit: Optional[str], rank: str):
        self.suit = suit
        self.rank = rank

    def __repr__(self):
        return f"{self.rank}{self.suit if self.suit else ''}"

    def to_str(self) -> str:
        return str(self)

    @classmethod
    def from_str(cls, card_str: str) -> 'Card':
        if not card_str: return None # Handle empty string if it occurs
        if card_str.upper() == JOKER_RANK:
            return cls(None, JOKER_RANK)
        if len(card_str) < 2: raise ValueError(f"Invalid card string format: {card_str}")
        rank = card_str[:-1].upper()
        suit = card_str[-1:].upper()
        if suit not in SUITS or rank not in RANKS: # Joker should be caught by the first condition
            raise ValueError(f"Invalid card string components: {card_str}")
        return cls(suit, rank)

    def get_value(self) -> int: # Value for discarding/absorbing damage
        if self.rank == JOKER_RANK: return 0
        if self.rank == "A": return 1
        if self.rank.isdigit(): return int(self.rank)
        if self.rank == "J": return 10
        if self.rank == "Q": return 15
        if self.rank == "K": return 20
        return 0

    def get_attack_power(self) -> int: # Attack power when played from hand
        if self.rank == "A": return 1
        if self.rank.isdigit(): return int(self.rank)
        if self.rank == "J": return 10
        if self.rank == "Q": return 15
        if self.rank == "K": return 20
        return 0 # Joker doesn't attack directly

# --- SQLAlchemy Setup ---
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("WARNING: DATABASE_URL environment variable not set. Using in-memory SQLite for now (not persistent).")
    DATABASE_URL = "sqlite:///:memory:" # Fallback for local testing if no DB_URL

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Database Models ---
class GameRoom(Base):
    __tablename__ = "game_rooms"

    room_code = Column(String, primary_key=True, index=True)
    status = Column(SQLAEnum(GameStatusEnum), default=GameStatusEnum.WAITING_FOR_PLAYERS)
    created_by_player_id = Column(String, nullable=False) # ID of the player who created the room
    current_player_idx = Column(Integer, default=0)
    player_order = Column(JSONB, default=list) # List of player_ids in turn order

    tavern_deck = Column(JSONB, default=list)  # List of card strings
    castle_deck = Column(JSONB, default=list)  # List of card strings
    hospital = Column(JSONB, default=list)     # List of card strings

    current_enemy_str = Column(String, nullable=True)
    current_enemy_health = Column(Integer, default=0)
    current_enemy_base_attack = Column(Integer, default=0) # Base attack, shield is subtracted from this
    current_enemy_shield = Column(Integer, default=0)

    active_joker_cancels_immunity = Column(Boolean, default=False)
    consecutive_yields = Column(Integer, default=0)
    solo_jokers_available = Column(Integer, default=0) # For solo mode

    damage_to_defend = Column(Integer, default=0) 
    player_to_defend_id = Column(String, nullable=True)
    jester_chooser_id = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    players = relationship("Player", back_populates="room", cascade="all, delete-orphan")

class Player(Base):
    __tablename__ = "players"

    id = Column(String, primary_key=True, index=True) # Globally unique player ID (e.g., from auth system or session)
    room_code = Column(String, ForeignKey("game_rooms.room_code"), nullable=False, index=True)
    player_name = Column(String, nullable=False)
    hand = Column(JSONB, default=list) # List of card strings

    room = relationship("GameRoom", back_populates="players")

    created_at = Column(DateTime(timezone=True), server_default=func.now())

def initialize_database():
    """Drops and recreates all tables in the database. WARNING: This deletes existing data."""
    # Drop all tables defined in Base.metadata (order might matter due to foreign keys)
    # For more complex scenarios, consider a migration tool like Alembic.
    # Dropping in reverse order of creation or explicitly handling dependencies is safer.
    # However, for this simple schema, dropping all should work if Player depends on GameRoom.
    # If GameRoom depends on Player (which it doesn't seem to directly for table structure),
    # the order would need to be reversed or handled by the DB.
    # Base.metadata.drop_all(bind=engine) # This drops all tables known to Base.metadata
    
    # A more robust way to handle potential foreign key constraints during drop:
    # Get tables in an order that respects dependencies for dropping.
    # This is a simplified approach; Alembic is better for complex cases.
    inspector = inspect(engine)
    # Drop dependent tables first (Player) then GameRoom
    if inspector.has_table(Player.__tablename__):
        Player.__table__.drop(engine)
    if inspector.has_table(GameRoom.__tablename__):
        GameRoom.__table__.drop(engine)
        
    Base.metadata.create_all(bind=engine)
    print("Database tables dropped and recreated.")

# --- Database Session Management ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Helper Functions for DB Interaction and Game State ---
def _serialize_deck(deck: List[Card]) -> List[str]:
    return [card.to_str() for card in deck]

def _deserialize_deck(deck_str: List[str]) -> List[Card]:
    return [Card.from_str(cs) for cs in deck_str if cs] # Ensure cs is not empty

def _get_game_room(db: Session, room_code: str) -> Optional[GameRoom]:
    return db.query(GameRoom).filter(GameRoom.room_code == room_code).first()

def _get_player(db: Session, player_id: str, room_code: str = None) -> Optional[Player]:
    query = db.query(Player).filter(Player.id == player_id)
    if room_code:
        query = query.filter(Player.room_code == room_code)
    return query.first()

def _get_current_player_obj(room: GameRoom, db: Session) -> Optional[Player]:
    if not room.player_order or room.current_player_idx >= len(room.player_order):
        return None
    current_player_id = room.player_order[room.current_player_idx]
    return _get_player(db, current_player_id, room.room_code)

def _create_standard_deck() -> List[Card]:
    return [Card(suit, rank) for suit in SUITS for rank in RANKS if rank not in ["J", "Q", "K"]]

def _create_castle_deck_cards() -> List[Card]:
    jacks = [Card(s, "J") for s in SUITS]
    queens = [Card(s, "Q") for s in SUITS]
    kings = [Card(s, "K") for s in SUITS]
    random.shuffle(jacks)
    random.shuffle(queens)
    random.shuffle(kings)
    return jacks + queens + kings

def _deal_cards_to_players(room: GameRoom, db: Session):
    num_players = len(room.players)
    if not num_players: return

    max_hand_size = HAND_SIZES_CONST.get(num_players, 5)
    tavern_deck_cards = _deserialize_deck(room.tavern_deck)

    for _ in range(max_hand_size): # Iterate for max hand size rounds
        for player_id_in_order in room.player_order:
            player_obj = _get_player(db, player_id_in_order, room.room_code)
            if player_obj:
                player_hand_cards = _deserialize_deck(player_obj.hand)
                if len(player_hand_cards) < max_hand_size and tavern_deck_cards:
                    card_to_deal = tavern_deck_cards.pop(0) # Take from top
                    player_hand_cards.append(card_to_deal)
                    player_obj.hand = _serialize_deck(player_hand_cards) # Update DB field
    room.tavern_deck = _serialize_deck(tavern_deck_cards) # Update DB field

def _assemble_game_state_dict(room: GameRoom, perspective_player_id: Optional[str] = None) -> Dict[str, Any]:
    if not room: return None

    players_info = []
    for p_obj in room.players: # Iterate over Player objects in the relationship
        player_hand_cards = _deserialize_deck(p_obj.hand)
        player_data = {"id": p_obj.id, "name": p_obj.player_name, "hand_size": len(player_hand_cards)}
        if perspective_player_id == p_obj.id:
            player_data["hand"] = [card.to_str() for card in player_hand_cards]
        players_info.append(player_data)

    current_enemy_card = Card.from_str(room.current_enemy_str) if room.current_enemy_str else None
    current_enemy_effective_attack = 0
    if current_enemy_card:
        current_enemy_effective_attack = max(0, room.current_enemy_base_attack - room.current_enemy_shield)


    return {
        "room_code": room.room_code,
        "created_by_player_id": room.created_by_player_id, # Add this line
        "status": room.status.name,
        "players": players_info, # Using fetched player objects
        "player_order": room.player_order,
        "tavern_deck_size": len(_deserialize_deck(room.tavern_deck)),
        "castle_deck_size": len(_deserialize_deck(room.castle_deck)),
        "hospital_size": len(_deserialize_deck(room.hospital)),
        "hospital_cards": room.hospital, # NEW: Add the list of card strings from the hospital
        "current_enemy": room.current_enemy_str,
        "current_enemy_health": room.current_enemy_health,
        "current_enemy_attack": current_enemy_effective_attack, # This is the effective attack
        "current_enemy_base_attack": room.current_enemy_base_attack, # Actual base attack of enemy
        "current_enemy_shield": room.current_enemy_shield,
        "current_player_id": room.player_order[room.current_player_idx] if room.player_order and room.current_player_idx < len(room.player_order) else None,
        "active_joker_cancels_immunity": room.active_joker_cancels_immunity,
        "consecutive_yields": room.consecutive_yields,
        "solo_jokers_available": room.solo_jokers_available if len(room.players) == 1 else None,
        "damage_to_defend": room.damage_to_defend,
        "player_to_defend_id": room.player_to_defend_id,
        "jester_chooser_id": room.jester_chooser_id,
    }

# --- Regicide Engine API Functions ---

def create_room(player_id: str, player_name: str) -> Tuple[Optional[str], Optional[str]]:
    db_gen = get_db()
    db = next(db_gen)
    try:
        room_code = str(uuid.uuid4())[:6].upper()
        while _get_game_room(db, room_code):
            room_code = str(uuid.uuid4())[:6].upper()

        new_room = GameRoom(room_code=room_code, created_by_player_id=player_id, status=GameStatusEnum.WAITING_FOR_PLAYERS)
        db.add(new_room)
        db.commit() # Commit room first to satisfy foreign key for player

        # Automatically join the creator
        success, error = join_room(room_code, player_id, player_name) # join_room will handle its own db session
        if not success:
            db.delete(new_room) # Clean up room if join fails (should be rare)
            db.commit()
            return None, f"Failed to add creator to room: {error}"

        return room_code, None
    except Exception as e:
        db.rollback()
        return None, f"Database error creating room: {str(e)}"
    finally:
        next(db_gen, None) # Close session


def join_room(room_code: str, player_id: str, player_name: str) -> Tuple[bool, Optional[str]]:
    db_gen = get_db()
    db = next(db_gen)
    try:
        room = _get_game_room(db, room_code)
        if not room:
            return False, "Room not found."
        if room.status != GameStatusEnum.WAITING_FOR_PLAYERS:
            return False, "Game has already started or finished."
        if len(room.players) >= MAX_PLAYERS_CONST:
            return False, "Room is full."

        existing_player = _get_player(db, player_id, room_code)
        if existing_player:
            # Optionally, update name or just confirm rejoining waiting room
            existing_player.player_name = player_name # Update name if rejoining
            # return True, "Player already in room. Details updated." # Or specific reconnect logic
        else:
            new_player = Player(id=player_id, room_code=room_code, player_name=player_name, hand=[])
            db.add(new_player)
        
        # Update player_order if not already there
        current_player_order = list(room.player_order) # Make a mutable copy
        if player_id not in current_player_order:
            current_player_order.append(player_id)
            room.player_order = current_player_order # Assign back to trigger SQLAlchemy change detection

        db.commit()
        return True, None
    except Exception as e:
        db.rollback()
        return False, f"Database error joining room: {str(e)}"
    finally:
        next(db_gen, None)

def get_game_state(room_code: str, perspective_player_id: Optional[str] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    db_gen = get_db()
    db = next(db_gen)
    try:
        room = _get_game_room(db, room_code)
        if not room:
            return None, "Room not found."
        return _assemble_game_state_dict(room, perspective_player_id), None
    except Exception as e:
        # No rollback needed for read operation
        return None, f"Error fetching game state: {str(e)}"
    finally:
        next(db_gen, None)

def start_game(room_code: str, requesting_player_id: str) -> Tuple[bool, Optional[str]]:
    db_gen = get_db()
    db = next(db_gen)
    try:
        room = _get_game_room(db, room_code)
        if not room: return False, "Room not found."
        if room.created_by_player_id != requesting_player_id: return False, "Only the room creator can start the game."
        if room.status != GameStatusEnum.WAITING_FOR_PLAYERS: return False, "Game already started or finished."
        if len(room.players) < MIN_PLAYERS_CONST: return False, f"Not enough players. Need at least {MIN_PLAYERS_CONST}."

        room.castle_deck = _serialize_deck(_create_castle_deck_cards())
        
        tavern_cards = _create_standard_deck()
        num_jokers_to_add = JOKERS_IN_DECK_CONST.get(len(room.players), 0)
        for _ in range(num_jokers_to_add):
            tavern_cards.append(Card(None, JOKER_RANK))
        random.shuffle(tavern_cards)
        room.tavern_deck = _serialize_deck(tavern_cards)
        
        room.hospital = [] # Clear hospital
        room.current_player_idx = 0 # Creator (usually first in order) starts
        random.shuffle(room.player_order) # Shuffle player order at game start

        # Solo mode Jokers
        if len(room.players) == 1:
            room.solo_jokers_available = 2
        else:
            room.solo_jokers_available = 0

        _deal_cards_to_players(room, db) # This will modify room.tavern_deck and player hands

        # Set up first enemy
        castle_deck_cards = _deserialize_deck(room.castle_deck)
        if not castle_deck_cards:
            room.status = GameStatusEnum.LOST # Or an error state
            db.commit()
            return False, "Error: Castle deck empty at start."

        first_enemy_card = castle_deck_cards.pop(0)
        room.castle_deck = _serialize_deck(castle_deck_cards) # Update deck
        room.current_enemy_str = first_enemy_card.to_str()
        room.current_enemy_health = ROYAL_HEALTH_CONST[first_enemy_card.rank]
        room.current_enemy_base_attack = ROYAL_ATTACK_CONST[first_enemy_card.rank]
        room.current_enemy_shield = 0
        room.active_joker_cancels_immunity = False
        room.consecutive_yields = 0
        room.status = GameStatusEnum.IN_PROGRESS
        
        db.commit()
        return True, None
    except Exception as e:
        db.rollback()
        return False, f"Database error starting game: {str(e)}"
    finally:
        next(db_gen, None)

def _advance_turn(room: GameRoom):
    room.current_player_idx = (room.current_player_idx + 1) % len(room.player_order)

def _handle_enemy_defeat(room: GameRoom, overkill: bool, defeated_enemy_card: Card, db: Session):
    hospital_cards = _deserialize_deck(room.hospital)
    if overkill:
        hospital_cards.append(defeated_enemy_card)
    else: # Exact kill
        tavern_deck_cards = _deserialize_deck(room.tavern_deck)
        tavern_deck_cards.insert(0, defeated_enemy_card) # Place on top
        room.tavern_deck = _serialize_deck(tavern_deck_cards)
    room.hospital = _serialize_deck(hospital_cards)

    room.active_joker_cancels_immunity = False # Reset joker effect

    castle_deck_cards = _deserialize_deck(room.castle_deck)
    if not castle_deck_cards:
        room.status = GameStatusEnum.WON
        room.current_enemy_str = None # No more enemies
        return

    # Set up next enemy
    next_enemy_card = castle_deck_cards.pop(0)
    room.castle_deck = _serialize_deck(castle_deck_cards)
    room.current_enemy_str = next_enemy_card.to_str()
    room.current_enemy_health = ROYAL_HEALTH_CONST[next_enemy_card.rank]
    room.current_enemy_base_attack = ROYAL_ATTACK_CONST[next_enemy_card.rank]
    room.current_enemy_shield = 0
    # Current player takes another turn, Royal does not attack this interim. Turn does not advance yet.

def play_cards(room_code: str, player_id: str, card_strs_played_from_req: List[str]) -> Tuple[bool, Optional[str]]:
    db_gen = get_db()
    db = next(db_gen)
    try:
        room = _get_game_room(db, room_code)
        if not room or room.status not in [GameStatusEnum.IN_PROGRESS]:
            return False, "Game not found or not in progress."

        player = _get_player(db, player_id, room_code)
        current_player_obj = _get_current_player_obj(room, db)
        if not player or player != current_player_obj:
            return False, "Not your turn or player not found in this room."
        if not room.current_enemy_str:
            return False, "No active enemy."

        current_enemy_card = Card.from_str(room.current_enemy_str)
        player_hand_cards = _deserialize_deck(player.hand)
        
        played_cards_obj: List[Card] = []
        temp_hand_for_validation = player_hand_cards[:] # Copy for validation

        for card_str_req in card_strs_played_from_req:
            card_to_play = Card.from_str(card_str_req)
            found_in_hand = False
            for i, hand_card in enumerate(temp_hand_for_validation):
                if hand_card.suit == card_to_play.suit and hand_card.rank == card_to_play.rank:
                    played_cards_obj.append(temp_hand_for_validation.pop(i))
                    found_in_hand = True
                    break
            if not found_in_hand:
                return False, f"Card {card_str_req} not found in hand or already selected."
        
        if not played_cards_obj: return False, "No cards were played."

        # --- Validate Play (same logic as before, using Card objects) ---
        is_joker_play = any(c.rank == JOKER_RANK for c in played_cards_obj)
        total_attack_value = 0

        if is_joker_play:
            if len(played_cards_obj) > 1: return False, "Joker must be played alone."
            joker_card = played_cards_obj[0]
            room.active_joker_cancels_immunity = True
            
            hospital_list = _deserialize_deck(room.hospital)
            hospital_list.append(joker_card)
            room.hospital = _serialize_deck(hospital_list)
            
            player.hand = _serialize_deck(temp_hand_for_validation) # Update player's hand
            
            room.status = GameStatusEnum.AWAITING_JESTER_CHOICE
            room.jester_chooser_id = player.id
            room.consecutive_yields = 0 # Reset yields as a play was made
            db.commit()
            return True, f"Joker played by {player.player_name}. Immunity cancelled for {current_enemy_card}. {player.player_name} must choose the next player."

        # Not a Joker play
        if len(played_cards_obj) == 1:
            total_attack_value = played_cards_obj[0].get_attack_power()
        elif len(played_cards_obj) == 2 and any(c.rank == "A" for c in played_cards_obj): # Animal Companion
            ace = next(c for c in played_cards_obj if c.rank == "A")
            other_card = next(c for c in played_cards_obj if c.rank != "A")
            if other_card.rank == JOKER_RANK: return False, "Ace cannot be companioned with a Joker."
            total_attack_value = other_card.get_attack_power() + ace.get_attack_power()
        else: # Set play
            first_rank = played_cards_obj[0].rank
            if not all(c.rank == first_rank for c in played_cards_obj): return False, "Set play cards must be of the same rank."
            if not first_rank.isdigit() or not (2 <= int(first_rank) <= 5): return False, "Set play only allowed for ranks 2, 3, 4, or 5."
            total_attack_value = sum(c.get_attack_power() for c in played_cards_obj)
            if total_attack_value > 10: return False, "Set play attack sum cannot exceed 10."

        player.hand = _serialize_deck(temp_hand_for_validation) # Commit hand changes
        room.consecutive_yields = 0

        # --- Activate Suit Powers ---
        actual_damage_to_deal = total_attack_value
        enemy_is_immune_to_suit = lambda suit: (suit == current_enemy_card.suit and not room.active_joker_cancels_immunity)

        tavern_deck_cards = _deserialize_deck(room.tavern_deck)
        hospital_cards = _deserialize_deck(room.hospital)

        for card_obj in played_cards_obj:
            if card_obj.suit == "H" and not enemy_is_immune_to_suit("H"):
                random.shuffle(hospital_cards)
                num_to_heal = min(total_attack_value, len(hospital_cards))
                healed_from_hospital = [hospital_cards.pop(0) for _ in range(num_to_heal)]
                tavern_deck_cards.extend(healed_from_hospital)
            elif card_obj.suit == "D" and not enemy_is_immune_to_suit("D"):
                drawn_this_turn = 0
                for i in range(len(room.player_order)):
                    if drawn_this_turn >= total_attack_value: break
                    p_idx_to_draw = (room.current_player_idx + i) % len(room.player_order)
                    player_to_draw_id = room.player_order[p_idx_to_draw]
                    player_to_draw_obj = _get_player(db, player_to_draw_id, room.room_code)
                    if player_to_draw_obj:
                        p_hand = _deserialize_deck(player_to_draw_obj.hand)
                        max_h_size = HAND_SIZES_CONST[len(room.players)]
                        while len(p_hand) < max_h_size and tavern_deck_cards and drawn_this_turn < total_attack_value:
                            p_hand.append(tavern_deck_cards.pop(0))
                            drawn_this_turn += 1
                        player_to_draw_obj.hand = _serialize_deck(p_hand)
            elif card_obj.suit == "S" and not enemy_is_immune_to_suit("S"):
                room.current_enemy_shield += total_attack_value
            elif card_obj.suit == "C" and not enemy_is_immune_to_suit("C"):
                actual_damage_to_deal *= 2 # Applied once

        room.tavern_deck = _serialize_deck(tavern_deck_cards) # Update from Diamond/Heart effects
        
        # Played cards go to hospital (after powers resolve)
        hospital_cards.extend(played_cards_obj)
        room.hospital = _serialize_deck(hospital_cards)


        # --- Deal Damage to Royal ---
        room.current_enemy_health -= actual_damage_to_deal
        
        if room.current_enemy_health <= 0:
            overkill = room.current_enemy_health < 0
            defeated_enemy_card_obj = Card.from_str(room.current_enemy_str) # Get it before it's cleared
            _handle_enemy_defeat(room, overkill, defeated_enemy_card_obj, db)
            db.commit()
            if room.status == GameStatusEnum.WON:
                return True, f"{player.player_name} defeated the final King! Players win!"
            return True, f"{player.player_name} dealt {actual_damage_to_deal} damage. Enemy {defeated_enemy_card_obj} defeated! New enemy: {room.current_enemy_str}. {player.player_name}'s turn again."
        else:
            # --- Royal Attacks ---
            royal_attack_power = max(0, room.current_enemy_base_attack - room.current_enemy_shield)
            
            room.status = GameStatusEnum.AWAITING_DEFENSE
            room.damage_to_defend = royal_attack_power
            room.player_to_defend_id = player.id
            # Turn does not advance here; it advances after successful defense or if game is lost.
            db.commit()
            return True, f"{player.player_name} dealt {actual_damage_to_deal} damage. {current_enemy_card} has {room.current_enemy_health} HP. {player_name} must defend against {royal_attack_power} damage."

    except ValueError as ve: # Catch card parsing errors etc.
        db.rollback()
        return False, f"Invalid card input or game data: {str(ve)}"
    except Exception as e:
        db.rollback()
        # Log the full error for debugging: print(f"Error in play_cards: {e}", file=sys.stderr)
        return False, f"An error occurred: {str(e)}"
    finally:
        next(db_gen, None)


def yield_turn(room_code: str, player_id: str) -> Tuple[bool, Optional[str]]:
    db_gen = get_db()
    db = next(db_gen)
    try:
        room = _get_game_room(db, room_code)
        if not room or room.status != GameStatusEnum.IN_PROGRESS:
            return False, "Game not found or not in progress."

        player = _get_player(db, player_id, room_code)
        current_player_obj = _get_current_player_obj(room, db)
        if not player or player != current_player_obj:
            return False, "Not your turn or player not found."
        if not room.current_enemy_str:
            return False, "No active enemy."

        current_enemy_card = Card.from_str(room.current_enemy_str)
        room.consecutive_yields += 1
        if room.consecutive_yields >= len(room.player_order):
            room.status = GameStatusEnum.LOST
            db.commit()
            return False, "All players yielded consecutively. Game Over."

        # --- Royal Attacks (same logic as play_cards) ---
        royal_attack_power = max(0, room.current_enemy_base_attack - room.current_enemy_shield)
        
        room.status = GameStatusEnum.AWAITING_DEFENSE
        room.damage_to_defend = royal_attack_power
        room.player_to_defend_id = player.id
        # Turn does not advance here.
        db.commit()
        return True, f"{player.player_name} yielded. {player.player_name} must defend against {royal_attack_power} damage."
        
    except Exception as e:
        db.rollback()
        return False, f"Error during yield: {str(e)}"
    finally:
        next(db_gen, None)

def use_solo_joker_power(room_code: str, player_id: str) -> Tuple[bool, Optional[str]]:
    db_gen = get_db()
    db = next(db_gen)
    try:
        room = _get_game_room(db, room_code)
        if not room or room.status != GameStatusEnum.IN_PROGRESS:
            return False, "Game not found or not in progress."
        if len(room.players) != 1: return False, "Solo Joker power only for solo games."

        player = _get_player(db, player_id, room_code)
        current_player_obj = _get_current_player_obj(room, db)
        if not player or player != current_player_obj: return False, "Not your turn or player not found."

        if room.solo_jokers_available <= 0: return False, "No solo Jokers available."

        room.solo_jokers_available -= 1
        
        player_hand_cards = _deserialize_deck(player.hand)
        hospital_cards = _deserialize_deck(room.hospital)
        hospital_cards.extend(player_hand_cards) # Discard entire hand to hospital
        room.hospital = _serialize_deck(hospital_cards)
        player.hand = [] # Clear hand

        # Draw back up to max hand size for solo
        new_hand_solo: List[Card] = []
        tavern_deck_cards = _deserialize_deck(room.tavern_deck)
        max_solo_hand = HAND_SIZES_CONST[1]
        while len(new_hand_solo) < max_solo_hand and tavern_deck_cards:
            new_hand_solo.append(tavern_deck_cards.pop(0))
        
        player.hand = _serialize_deck(new_hand_solo)
        room.tavern_deck = _serialize_deck(tavern_deck_cards)
        
        db.commit()
        return True, f"{player.player_name} used a Solo Joker. Hand refreshed. {room.solo_jokers_available} solo Jokers remaining."
    except Exception as e:
        db.rollback()
        return False, f"Error using solo joker: {str(e)}"
    finally:
        next(db_gen, None)

def defend_against_attack(room_code: str, player_id: str, card_strs_to_discard: List[str]) -> Tuple[bool, Optional[str]]:
    db_gen = get_db()
    db = next(db_gen)
    try:
        room = _get_game_room(db, room_code)
        if not room or room.status != GameStatusEnum.AWAITING_DEFENSE:
            return False, "Game not in AWAITING_DEFENSE state or room not found."

        if room.player_to_defend_id != player_id:
            return False, "Not your turn to defend."

        player = _get_player(db, player_id, room_code)
        if not player:
            return False, "Player not found in this room."

        player_hand_cards = _deserialize_deck(player.hand)
        cards_to_discard_obj: List[Card] = []
        temp_hand_for_validation = player_hand_cards[:] # Copy for validation

        for card_str_req in card_strs_to_discard:
            card_to_discard = Card.from_str(card_str_req)
            found_in_hand = False
            for i, hand_card in enumerate(temp_hand_for_validation):
                if hand_card.suit == card_to_discard.suit and hand_card.rank == card_to_discard.rank:
                    cards_to_discard_obj.append(temp_hand_for_validation.pop(i))
                    found_in_hand = True
                    break
            if not found_in_hand:
                db.rollback() # Important: rollback if cards are invalid
                return False, f"Card {card_str_req} not found in hand or already selected for discard."
        
        defense_value = sum(c.get_value() for c in cards_to_discard_obj)

        if defense_value < room.damage_to_defend:
            # Player chose cards but they are not enough. This is a loss.
            # The rules state: "If a player cannot discard enough cards to satisfy the full damage amount, that player is eliminated, and all players lose the game immediately."
            # This implies if they *can* but *choose* not to discard enough, it's also a loss if their choice is insufficient.
            # For simplicity, we assume if they submit cards, they believe it's enough. If not, it's a loss.
            # A more complex UI might allow them to re-select if their current selection is < required.
            # Here, we assume the submitted cards are their final choice for this attempt.
            room.status = GameStatusEnum.LOST
            # Optionally, move the chosen (insufficient) cards to hospital anyway
            player.hand = _serialize_deck(temp_hand_for_validation)
            hospital_cards = _deserialize_deck(room.hospital)
            hospital_cards.extend(cards_to_discard_obj)
            room.hospital = _serialize_deck(hospital_cards)
            db.commit()
            return False, f"Player {player.player_name} failed to discard enough value ({defense_value}/{room.damage_to_defend}). Game Over."

        # Sufficient defense
        player.hand = _serialize_deck(temp_hand_for_validation) # Update hand
        hospital_cards = _deserialize_deck(room.hospital)
        hospital_cards.extend(cards_to_discard_obj)
        room.hospital = _serialize_deck(hospital_cards)

        room.damage_to_defend = 0
        room.player_to_defend_id = None
        room.status = GameStatusEnum.IN_PROGRESS
        _advance_turn(room)
        
        db.commit()
        discarded_names = ", ".join([c.to_str() for c in cards_to_discard_obj])
        return True, f"{player.player_name} successfully defended by discarding {discarded_names}. Next player's turn."

    except ValueError as ve:
        db.rollback()
        return False, f"Invalid card input: {str(ve)}"
    except Exception as e:
        db.rollback()
        return False, f"Error during defense: {str(e)}"
    finally:
        next(db_gen, None)

def choose_next_player_after_jester(room_code: str, jester_player_id: str, chosen_next_player_id: str) -> Tuple[bool, Optional[str]]:
    db_gen = get_db()
    db = next(db_gen)
    try:
        room = _get_game_room(db, room_code)
        if not room or room.status != GameStatusEnum.AWAITING_JESTER_CHOICE:
            return False, "Game not awaiting Jester choice or room not found."

        if room.jester_chooser_id != jester_player_id:
            return False, "It's not your turn to choose the next player."

        chosen_player_obj = _get_player(db, chosen_next_player_id, room_code)
        if not chosen_player_obj:
            return False, f"Chosen player {chosen_next_player_id} not found in this room."

        try:
            next_player_idx = room.player_order.index(chosen_next_player_id)
        except ValueError:
            return False, f"Chosen player {chosen_next_player_id} is not in the player order."

        room.current_player_idx = next_player_idx
        room.jester_chooser_id = None
        room.status = GameStatusEnum.IN_PROGRESS
        
        db.commit()
        return True, f"{chosen_player_obj.player_name}'s turn has been set by {jester_player_id}."
    except Exception as e:
        db.rollback()
        return False, f"Error choosing next player: {str(e)}"
    finally:
        next(db_gen, None)

# Example of how to run initialize_database() (e.g., in a manage.py or one-off script)
if __name__ == "__main__":
    # This check prevents accidental execution if imported.
    # You would typically call initialize_database() from a separate management script
    # or carefully during application startup in a controlled way.
    # For example:
    # if input("Initialize database? (THIS WILL CREATE TABLES) (yes/no): ").lower() == 'yes':
    #     initialize_database()

    # Basic test after DB init (requires DATABASE_URL to be set up)
    # This is just for rudimentary local testing of the engine, not for production.
    # print("Running basic engine tests...")
    # initialize_database() # Ensure tables exist for this test sequence
    #
    # p_id1 = "test_player_001"
    # p_name1 = "Alice"
    # p_id2 = "test_player_002"
    # p_name2 = "Bob"
    #
    # rc, err = create_room(p_id1, p_name1)
    # if err: print(f"Create room error: {err}")
    # else: print(f"Room created: {rc}")
    #
    # if rc:
    #     s, err = join_room(rc, p_id2, p_name2)
    #     if err: print(f"Join room error: {err}")
    #     else: print(f"Bob joined: {s}")
    #
    #     gs, err = get_game_state(rc, p_id1)
    #     if err: print(f"Get state error: {err}")
    #     else: print(f"Initial Game State for Alice: {gs}")
    #
    #     s, err = start_game(rc, p_id1)
    #     if err: print(f"Start game error: {err}")
    #     else: print(f"Game started: {s}")
    #
    #     gs_after_start, err = get_game_state(rc, p_id1)
    #     if err: print(f"Get state after start error: {err}")
    #     else:
    #         print(f"Game State for Alice after start: {gs_after_start}")
    #         current_player_id_test = gs_after_start.get("current_player_id")
    #         current_player_hand_test = []
    #         for p_info in gs_after_start.get("players", []):
    #             if p_info["id"] == current_player_id_test:
    #                 current_player_hand_test = p_info.get("hand", [])
    #                 break
    #         print(f"Current player {current_player_id_test} hand: {current_player_hand_test}")
    #
    #         # Example: Play first card from current player's hand
    #         if current_player_id_test and current_player_hand_test:
    #             card_to_play_str = current_player_hand_test[0]
    #             print(f"Attempting to play: {card_to_play_str} by {current_player_id_test}")
    #             s_play, msg_play = play_cards(rc, current_player_id_test, [card_to_play_str])
    #             print(f"Play card result: {s_play}, Message: {msg_play}")
    #
    #             gs_after_play, _ = get_game_state(rc, p_id1)
    #             print(f"Game State after play: {gs_after_play}")

    pass # Keep the __main__ block minimal or for specific setup tasks