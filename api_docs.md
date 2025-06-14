# Regicide Game API Documentation

This document provides details for interacting with the Regicide Game API.

**Base URL:** `[YOUR_API_BASE_URL]` (e.g., `https://your-regicide-api.onrender.com/api`)

**Authentication:** There is no explicit authentication in this version. Player identification is managed by a `player_id` sent in request bodies.

**Content-Type:** All request bodies should be `application/json`. All responses will be `application/json`.

## Standard Response Format

**Successful Responses (2xx status codes):**
```json
{
  "status": "success",
  "data": { /* Request-specific data object */ },
  "message": "Descriptive success message (optional)"
}
```

**Error Responses (4xx or 5xx status codes):**
```json
{
  "status": "error",
  "message": "Descriptive error message",
  "error_code": "OPTIONAL_ERROR_CODE_STRING (e.g., 'ROOM_FULL')" // Optional
}
```

## Endpoints

---

### 1. Health Check

*   **Endpoint:** `/health`
*   **Method:** `GET`
*   **Description:** Checks if the API is running and healthy.
*   **Request Body:** None
*   **Success Response (200 OK):**
    ```json
    {
      "status": "success",
      "data": null,
      "message": "API is healthy"
    }
    ```

---

### 2. Create Room

*   **Endpoint:** `/create_room`
*   **Method:** `POST`
*   **Description:** Creates a new game room and automatically adds the creator as a player.
*   **Request Body:**
    ```json
    {
      "player_id": "string (unique ID for the player)",
      "player_name": "string (display name for the player)"
    }
    ```
*   **Success Response (201 Created):**
    ```json
    {
      "status": "success",
      "data": {
        "room_code": "string (6-character unique room code)",
        "player_id": "string (ID of the player who created the room)"
      },
      "message": "Room created successfully."
    }
    ```
*   **Error Responses:**
    *   `400 Bad Request`: If `player_id` or `player_name` is missing, or other internal error.

---

### 3. Join Room

*   **Endpoint:** `/join_room`
*   **Method:** `POST`
*   **Description:** Allows a player to join an existing game room that is waiting for players.
*   **Request Body:**
    ```json
    {
      "room_code": "string (the 6-character room code)",
      "player_id": "string (unique ID for the joining player)",
      "player_name": "string (display name for the joining player)"
    }
    ```
*   **Success Response (200 OK):**
    Returns the current game state object (see "Game State Object" section below).
    ```json
    {
      "status": "success",
      "data": { /* Game State Object */ },
      "message": "Joined room successfully."
    }
    ```
*   **Error Responses:**
    *   `400 Bad Request`: If required fields are missing.
    *   `404 Not Found`: If `room_code` does not exist.
    *   `400 Bad Request` (or custom): If room is full or game has already started.

---

### 4. Get Game State

*   **Endpoint:** `/game_state/{room_code}`
*   **Method:** `GET`
*   **Description:** Retrieves the current state of a specific game room.
*   **URL Parameters:**
    *   `room_code`: The 6-character code of the room.
*   **Query Parameters (Optional):**
    *   `player_id`: `string` - If provided, the response will include the hand details for this specific player.
*   **Request Body:** None
*   **Success Response (200 OK):**
    Returns the game state object (see "Game State Object" section below).
    ```json
    {
      "status": "success",
      "data": { /* Game State Object */ },
      "message": "Success"
    }
    ```
*   **Error Responses:**
    *   `404 Not Found`: If `room_code` does not exist.

---

### 5. Start Game

*   **Endpoint:** `/start_game`
*   **Method:** `POST`
*   **Description:** Starts the game in a room that is `WAITING_FOR_PLAYERS`. Only the player who created the room can start it. This initializes decks, deals cards, and sets the first enemy.
*   **Request Body:**
    ```json
    {
      "room_code": "string (the 6-character room code)",
      "player_id": "string (ID of the player attempting to start the game; must be the creator)"
    }
    ```
*   **Success Response (200 OK):**
    Returns the updated game state object (status will be `IN_PROGRESS`).
    ```json
    {
      "status": "success",
      "data": { /* Game State Object */ },
      "message": "Game started successfully."
    }
    ```
*   **Error Responses:**
    *   `400 Bad Request`: If room not found, game already started/finished, not enough players.
    *   `403 Forbidden`: If `player_id` is not the room creator.

---

### 6. Play Cards

*   **Endpoint:** `/play_cards`
*   **Method:** `POST`
*   **Description:** Allows the current player to play one or more cards from their hand to attack the enemy or use card powers. If the enemy is not defeated and attacks, the game state will transition to `AWAITING_DEFENSE`. If a Jester is played, the game state will transition to `AWAITING_JESTER_CHOICE`.
*   **Request Body:**
    ```json
    {
      "room_code": "string",
      "player_id": "string (ID of the current player)",
      "cards": ["string", "string", ...] // List of card strings (e.g., "H5", "SA", "X" for Joker)
    }
    ```
*   **Success Response (200 OK):**
    Returns the updated game state object along with a message about the action's outcome.
    ```json
    {
      "status": "success",
      "data": { /* Game State Object */ },
      "message": "Action message (e.g., 'Cards played successfully.', 'Enemy defeated!', 'Player must defend.', 'Jester played, choose next player.', 'Game Over!')"
    }
    ```
*   **Error Responses:**
    *   `400 Bad Request`: Invalid play (e.g., card not in hand, not player's turn, invalid card combination, game not in progress). Returns current game state in `data` field if possible.
    *   If the game ends due to this action, the message will indicate win/loss, and status in game state will be updated.

---

### 7. Yield Turn

*   **Endpoint:** `/yield_turn`
*   **Method:** `POST`
*   **Description:** Allows the current player to yield their turn. The enemy will attack, and the game state will transition to `AWAITING_DEFENSE`.
*   **Request Body:**
    ```json
    {
      "room_code": "string",
      "player_id": "string (ID of the current player)"
    }
    ```
*   **Success Response (200 OK):**
    Returns the updated game state object with a message.
    ```json
    {
      "status": "success",
      "data": { /* Game State Object */ },
      "message": "Action message (e.g., 'Turn yielded. Player must defend against X damage.')"
    }
    ```
*   **Error Responses:**
    *   `400 Bad Request`: Not player's turn, game not in progress, cannot yield.
    *   If all players yield consecutively, game ends.

---

### 8. Defend Against Attack

*   **Endpoint:** `/defend`
*   **Method:** `POST`
*   **Description:** Allows the player designated by `player_to_defend_id` to discard cards from their hand to absorb damage. This is called when game status is `AWAITING_DEFENSE`.
*   **Request Body:**
    ```json
    {
      "room_code": "string",
      "player_id": "string (ID of the player who needs to defend)",
      "cards": ["string", "string", ...] // List of card strings to discard for defense
    }
    ```
*   **Success Response (200 OK):**
    Returns the updated game state object. If defense is successful, status becomes `IN_PROGRESS` and turn advances. If defense fails, status becomes `LOST`.
    ```json
    {
      "status": "success",
      "data": { /* Game State Object */ },
      "message": "Defense successful. Next player's turn." // or "Failed to defend. Game Over."
    }
    ```
*   **Error Responses:**
    *   `400 Bad Request`: Not in `AWAITING_DEFENSE` state, wrong player, cards not in hand, insufficient discard value leading to game loss.

---

### 9. Choose Next Player (After Jester)

*   **Endpoint:** `/choose_next_player`
*   **Method:** `POST`
*   **Description:** Allows the player who played a Jester (identified by `jester_chooser_id`) to choose the next player. This is called when game status is `AWAITING_JESTER_CHOICE`.
*   **Request Body:**
    ```json
    {
      "room_code": "string",
      "player_id": "string (ID of the player who played the Jester)",
      "chosen_player_id": "string (ID of the player to take the next turn)"
    }
    ```
*   **Success Response (200 OK):**
    Returns the updated game state object. Status becomes `IN_PROGRESS` and `current_player_id` is updated.
    ```json
    {
      "status": "success",
      "data": { /* Game State Object */ },
      "message": "Next player chosen successfully. [Chosen Player Name]'s turn."
    }
    ```
*   **Error Responses:**
    *   `400 Bad Request`: Not in `AWAITING_JESTER_CHOICE` state, wrong player choosing, invalid chosen player.

---

### 10. Use Solo Joker Power (Solo Mode Only)

*   **Endpoint:** `/use_solo_joker`
*   **Method:** `POST`
*   **Description:** In a solo game, allows the player to use one of their available solo joker powers (discard hand and redraw).
*   **Request Body:**
    ```json
    {
      "room_code": "string",
      "player_id": "string (ID of the solo player)"
    }
    ```
*   **Success Response (200 OK):**
    Returns the updated game state object with a message.
    ```json
    {
      "status": "success",
      "data": { /* Game State Object */ },
      "message": "Solo Joker power used. Hand refreshed..."
    }
    ```
*   **Error Responses:**
    *   `400 Bad Request`: Not a solo game, no solo jokers available, not player's turn.

---

## Game State Object

The game state object is returned by several endpoints and represents the current status of the game.

```json
{
  "room_code": "string",
  "created_by_player_id": "string (ID of the player who created the room)", // If available
  "status": "string (Enum: WAITING_FOR_PLAYERS, IN_PROGRESS, AWAITING_DEFENSE, AWAITING_JESTER_CHOICE, WON, LOST)",
  "players": [
    {
      "id": "string",
      "name": "string",
      "hand_size": "integer",
      "hand": ["string", ...] // Only present if 'player_id' query param matches this player's ID during /game_state GET
    }
    // ... more players
  ],
  "player_order": ["string (player_id)", ...], // Order of turns
  "tavern_deck_size": "integer",
  "castle_deck_size": "integer",
  "hospital_size": "integer (discard pile)",
  "hospital_cards": ["string", ...], // NEW: List of cards in the discard pile (e.g., ["H5", "S10", "CA"])
  "current_enemy": "string (e.g., 'SJ' for Jack of Spades, null if no enemy)",
  "current_enemy_health": "integer",
  "current_enemy_attack": "integer (effective attack after shield)",
  "current_enemy_base_attack": "integer (base attack of the enemy before shield)",
  "current_enemy_shield": "integer (damage reduction from Spades)",
  "current_player_id": "string (ID of the player whose turn it is, null if game not started/over or if an action is pending from another player)",
  "player_to_defend_id": "string (ID of player who needs to discard cards for defense, null otherwise)",
  "damage_to_defend": "integer (Amount of damage the player_to_defend_id needs to absorb, 0 otherwise)",
  "jester_chooser_id": "string (ID of player who played a Jester and needs to choose next player, null otherwise)",
  "active_joker_cancels_immunity": "boolean",
  "consecutive_yields": "integer",
  "solo_jokers_available": "integer (or null if not a solo game)",
  "action_message": "string (Optional: a message describing the result of the last action)" // Added by some action endpoints
}
```

## Card String Format

*   **Suits:** H (Hearts), D (Diamonds), S (Spades), C (Clubs)
*   **Ranks:** A (Ace), 2, 3, 4, 5, 6, 7, 8, 9, 10, J (Jack), Q (Queen), K (King)
*   **Joker:** X
*   **Format:** Generally `RankSuit` (e.g., "AS" for Ace of Spades, "10D" for 10 of Diamonds, "HJ" for Jack of Hearts). "X" for Joker.

---
