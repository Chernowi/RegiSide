import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.exceptions import NotFound

# Import your models and engine functions
import regicide_engine as engine
from regicide_engine import initialize_database
# Removed imports for get_player_by_token, create_player, get_game, 
# create_game_db, update_game_db, get_game_state_for_player, Game, Player
# as they are not defined in regicide_engine.py or not used directly in this file.

# Call initialize_database() when the app module is loaded.
# This will ensure tables are created if they don't exist when the app starts.
initialize_database()

app = Flask(__name__)

# --- CORS Configuration ---
# It's crucial to set NETLIFY_FRONTEND_URL in your Render environment variables.
# For local development, you might run your frontend on http://localhost:3000, http://localhost:5173 (Vite), etc.
# Add your local development frontend URL to the list of origins if needed.
NETLIFY_APP_URL = os.environ.get('NETLIFY_FRONTEND_URL', 'http://localhost:3000') # Default for local dev

# Allow specific origins.
# If your Netlify URL might change or you have multiple preview deploys,
# you might need a more dynamic way to set this or use a wildcard (less secure).
# For production, be as specific as possible.
allowed_origins = [NETLIFY_APP_URL]
if "localhost" not in NETLIFY_APP_URL: # Add common local dev ports if not explicitly the NETLIFY_APP_URL
    allowed_origins.append("http://localhost:3000") # Common React dev port
    allowed_origins.append("http://localhost:5173") # Common Vite dev port
    allowed_origins.append("http://localhost:8080") # Common Vue dev port

CORS(app, resources={r"/api/*": {"origins": allowed_origins}}, supports_credentials=True)

# --- Database Initialization (Conceptual) ---
# Your regicide_engine.py should ideally handle its own database connection
# using the DATABASE_URL environment variable.
# If it needs an explicit init call, you could do it here:
# DATABASE_URL = os.environ.get('DATABASE_URL')
# if DATABASE_URL:
#     engine.initialize_database(DATABASE_URL)
# else:
#     if os.environ.get('FLASK_ENV') == 'production' or os.environ.get('RENDER'):
#         print("WARNING: DATABASE_URL not set in a production-like environment!")
#     else:
#         print("DATABASE_URL not set. Engine might use in-memory store or fail if DB is required.")

# --- Helper for Standard Responses ---
def success_response(data, message="Success", status_code=200):
    return jsonify({"status": "success", "data": data, "message": message}), status_code

def error_response(message, status_code=400, error_code=None, data=None): # Added 'data=None'
    response = {"status": "error", "message": message}
    if error_code:
        response["error_code"] = error_code
    if data: # Include data in the response if provided
        response["data"] = data
    return jsonify(response), status_code

# --- API Routes ---

@app.route('/api/health', methods=['GET'])
def health_check():
    """A simple health check endpoint."""
    return success_response(None, "API is healthy")

@app.route('/api/create_room', methods=['POST'])
def route_create_room():
    data = request.json
    player_id = data.get('player_id')
    player_name = data.get('player_name')
    custom_room_code = data.get('custom_room_code') # New: Get custom room code

    if not all([player_id, player_name]):
        return error_response("player_id and player_name are required.", 400)

    room_code, error = engine.create_room(player_id, player_name, custom_room_code=custom_room_code) # Pass it to the engine
    if error:
        return error_response(error, 400) # Or a more specific error code
    return success_response({"room_code": room_code, "player_id": player_id}, "Room created successfully.", 201)

@app.route('/api/join_room', methods=['POST'])
def route_join_room():
    data = request.json
    room_code = data.get('room_code')
    player_id = data.get('player_id')
    player_name = data.get('player_name')

    if not all([room_code, player_id, player_name]):
        return error_response("room_code, player_id, and player_name are required.", 400)

    success, error = engine.join_room(room_code, player_id, player_name)
    if error:
        return error_response(error, 400) # Could be 404 if room not found, 403 if full etc.

    # After joining, return the full game state
    game_state, state_error = engine.get_game_state(room_code, perspective_player_id=player_id)
    if state_error:
        return error_response(f"Joined room, but failed to fetch game state: {state_error}", 500)
    return success_response(game_state, "Joined room successfully.")


@app.route('/api/game_state/<room_code>', methods=['GET'])
def route_get_game_state(room_code):
    perspective_player_id = request.args.get('player_id') # Optional
    state, error = engine.get_game_state(room_code, perspective_player_id)
    if error:
        return error_response(error, 404 if "not found" in error.lower() else 400)
    return success_response(state)

@app.route('/api/start_game', methods=['POST'])
def route_start_game():
    data = request.json
    room_code = data.get('room_code')
    requesting_player_id = data.get('player_id') # ID of player attempting to start

    if not all([room_code, requesting_player_id]):
        return error_response("room_code and player_id are required.", 400)

    success, error = engine.start_game(room_code, requesting_player_id)
    if error:
        # Check for specific errors to return more appropriate status codes
        if "not creator" in error.lower() or "only the room creator" in error.lower():
            return error_response(error, 403) # Forbidden
        return error_response(error, 400)

    game_state, state_error = engine.get_game_state(room_code, perspective_player_id=requesting_player_id)
    if state_error:
        return error_response(f"Game started, but failed to fetch game state: {state_error}", 500)
    return success_response(game_state, "Game started successfully.")

@app.route('/api/play_cards', methods=['POST'])
def route_play_cards():
    data = request.json
    if not data: # Check if request.json is None
        return error_response("Request body must be valid JSON and Content-Type header must be application/json.", 400)
    room_code = data.get('room_code')
    player_id = data.get('player_id')
    cards_played_str = data.get('cards') # Expects a list of card strings e.g., ["H5", "S2"]

    if not all([room_code, player_id, isinstance(cards_played_str, list)]):
        return error_response("room_code, player_id, and a list of cards are required.", 400)

    action_successful, message = engine.play_cards(room_code, player_id, cards_played_str)
    
    # Always fetch the latest game state to return
    current_game_state, state_error = engine.get_game_state(room_code, perspective_player_id=player_id)
    if state_error:
         # If game ended, state_error might be "game not found" if it was cleaned up
        if action_successful and ("Game Over" in message or "win" in message or "lost" in message):
             return success_response({"game_over_message": message}, message) # Send specific game over message
        return error_response(f"Action processed with message '{message}', but failed to fetch updated game state: {state_error}", 500)

    if not action_successful and "Game Over" not in message and "win" not in message and "lost" not in message :
        # Action failed for a reason other than game ending
        return error_response(message, 400, data=current_game_state) # Now correctly handled

    return success_response(current_game_state, message)


@app.route('/api/yield_turn', methods=['POST'])
def route_yield_turn():
    data = request.json
    if not data: # Check if request.json is None
        return error_response("Request body must be valid JSON and Content-Type header must be application/json.", 400)
    room_code = data.get('room_code')
    player_id = data.get('player_id')

    if not all([room_code, player_id]):
        return error_response("room_code and player_id are required.", 400)

    action_successful, message = engine.yield_turn(room_code, player_id)

    current_game_state, state_error = engine.get_game_state(room_code, perspective_player_id=player_id)
    if state_error:
        if action_successful and ("Game Over" in message or "win" in message or "lost" in message):
             return success_response({"game_over_message": message}, message)
        return error_response(f"Action processed with message '{message}', but failed to fetch updated game state: {state_error}", 500)

    if not action_successful and "Game Over" not in message and "win" not in message and "lost" not in message:
        return error_response(message, 400, data=current_game_state) # Now correctly handled

    return success_response(current_game_state, message)

@app.route('/api/defend', methods=['POST'])
def route_defend():
    data = request.json
    if not data: # Check if request.json is None
        return error_response("Request body must be valid JSON and Content-Type header must be application/json.", 400)

    room_code = data.get('room_code')
    player_id = data.get('player_id')
    cards_to_discard = data.get('cards') # List of card strings

    if not all([room_code, player_id, isinstance(cards_to_discard, list)]):
        return error_response("room_code, player_id, and a list of cards are required.", 400)

    action_successful, message = engine.defend_against_attack(room_code, player_id, cards_to_discard)

    current_game_state, state_error = engine.get_game_state(room_code, perspective_player_id=player_id)
    if state_error:
        if action_successful and ("Game Over" in message or "win" in message or "lost" in message): # Game might have ended
             return success_response({"game_over_message": message, "final_state_fetch_error": state_error}, message)
        return error_response(f"Action processed with message '{message}', but failed to fetch updated game state: {state_error}", 500)

    if not action_successful:
         # If game ended due to failed defense, message will indicate "Game Over"
        return error_response(message, 400, data=current_game_state) # Now correctly handled

    return success_response(current_game_state, message)

@app.route('/api/choose_next_player', methods=['POST'])
def route_choose_next_player():
    data = request.json
    if not data: # Check if request.json is None
        return error_response("Request body must be valid JSON and Content-Type header must be application/json.", 400)
    room_code = data.get('room_code')
    jester_player_id = data.get('player_id') # The player who played the Jester
    chosen_next_player_id = data.get('chosen_player_id')

    if not all([room_code, jester_player_id, chosen_next_player_id]):
        return error_response("room_code, player_id (of jester player), and chosen_player_id are required.", 400)

    action_successful, message = engine.choose_next_player_after_jester(room_code, jester_player_id, chosen_next_player_id)

    # Fetch state from perspective of the jester player or the newly chosen player?
    # For consistency, let's use jester_player_id for now.
    current_game_state, state_error = engine.get_game_state(room_code, perspective_player_id=jester_player_id)
    if state_error:
        return error_response(f"Action processed with message '{message}', but failed to fetch updated game state: {state_error}", 500)

    if not action_successful:
        return error_response(message, 400, data=current_game_state) # Now correctly handled

    return success_response(current_game_state, message)

@app.route('/api/use_solo_joker', methods=['POST'])
def route_use_solo_joker():
    data = request.json
    if not data: # Check if request.json is None
        return error_response("Request body must be valid JSON and Content-Type header must be application/json.", 400)
    room_code = data.get('room_code')
    player_id = data.get('player_id')

    if not all([room_code, player_id]):
        return error_response("room_code and player_id are required.", 400)

    action_successful, message = engine.use_solo_joker_power(room_code, player_id) # Ensure this function exists in engine

    current_game_state, state_error = engine.get_game_state(room_code, perspective_player_id=player_id)
    if state_error:
        return error_response(f"Action processed with message '{message}', but failed to fetch updated game state: {state_error}", 500)

    if not action_successful:
        return error_response(message, 400, data=current_game_state) # Now correctly handled

    return success_response(current_game_state, message)


# --- Main Execution ---
if __name__ == '__main__':
    # Port is set by Render environment variable. Default to 5001 for local dev.
    port = int(os.environ.get('PORT', 5001))
    # Gunicorn will run this on Render, but this is useful for local `python app.py`
    # Set FLASK_ENV=development for debug mode locally, Render sets it to production
    debug_mode = os.environ.get('FLASK_ENV') == 'development' or os.environ.get('FLASK_DEBUG') == '1'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)