# RegiSide 🃏👑

A Flask-based REST API for the cooperative card game **Regicide**. RegiSide provides a backend server that allows 1-4 players to play Regicide cooperatively through HTTP endpoints.

## 🎮 About Regicide

Regicide is a cooperative, fantasy-themed card game where players work together to defeat 12 powerful enemies (Jacks, Queens, and Kings) from a standard 52-card deck. Players must use tactical decisions and careful hand management to overcome increasingly difficult royal enemies.

**Objective:** Defeat all 12 Royalty cards (4 Jacks, then 4 Queens, then 4 Kings) cooperatively. If any player is eliminated or cannot make a valid move, all players lose.

## 🚀 Features

- **Multiplayer Support**: 1-4 players can join and play cooperatively
- **Room System**: Create and join game rooms with unique codes
- **Real-time Game State**: Track game progress, player hands, and enemy status
- **Complete Rule Implementation**: Full Regicide mechanics including:
  - Animal Companions (Aces) with special abilities
  - Suit powers (Hearts heal, Diamonds draw, Spades double damage, Clubs heal all)
  - Jester cards for flexible gameplay
  - Defense mechanics against enemy attacks
- **RESTful API**: Clean HTTP endpoints for all game actions
- **PostgreSQL Support**: Persistent game state storage
- **CORS Enabled**: Ready for frontend integration

## 🛠️ Technology Stack

- **Backend**: Python 3.x + Flask
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Deployment**: Render-ready with Procfile
- **CORS**: Flask-CORS for frontend integration

## 📋 Prerequisites

- Python 3.7+
- PostgreSQL database
- pip package manager

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Chernowi/RegiSide.git
cd RegiSide
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables
```bash
# Database connection
export DATABASE_URL="postgresql://user:password@localhost:5432/regicide_db"

# Frontend URL for CORS (optional, defaults to localhost:3000)
export NETLIFY_FRONTEND_URL="https://your-frontend-url.netlify.app"
```

### 4. Run the Application
```bash
# Development
python app.py

# Production (using Gunicorn)
gunicorn app:app
```

The API will be available at `http://localhost:5000/api`

## 📚 API Documentation

### Base URL
```
http://localhost:5000/api
```

### Key Endpoints

#### Health Check
```http
GET /api/health
```

#### Create Room
```http
POST /api/create_room
Content-Type: application/json

{
  "player_name": "Alice",
  "max_players": 4
}
```

#### Join Room
```http
POST /api/join_room
Content-Type: application/json

{
  "room_code": "ABCD",
  "player_name": "Bob"
}
```

#### Start Game
```http
POST /api/start_game
Content-Type: application/json

{
  "room_code": "ABCD",
  "player_id": "player_uuid"
}
```

#### Play Card
```http
POST /api/play_card
Content-Type: application/json

{
  "room_code": "ABCD",
  "player_id": "player_uuid",
  "cards": ["AH", "2S"],
  "combo_type": "basic_attack"
}
```

#### Get Game State
```http
POST /api/get_game_state
Content-Type: application/json

{
  "room_code": "ABCD",
  "player_id": "player_uuid"
}
```

For complete API documentation, see [`api_docs.md`](./api_docs.md).

## 🎯 Game Rules

Detailed game rules and mechanics are available in [`regicide_rules.md`](./regicide_rules.md).

### Quick Overview
- **Setup**: 12 Royal enemies (4J, 4Q, 4K) form the Castle deck
- **Hand Management**: Players have limited hand sizes (5-8 cards based on player count)
- **Cooperative Play**: All players win together or lose together
- **Card Powers**: Each suit has special abilities when played
- **Enemy Scaling**: Jacks (20 HP/10 ATK) → Queens (30 HP/15 ATK) → Kings (40 HP/20 ATK)

## 🏗️ Project Structure

```
RegiSide/
├── app.py                 # Flask application and API routes
├── regicide_engine.py     # Core game logic and database models
├── regicide_rules.md      # Complete game rules documentation
├── api_docs.md           # Detailed API documentation
├── requirements.txt      # Python dependencies
├── Procfile             # Render deployment configuration
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## 🌐 Deployment

### Render Deployment
1. Connect your GitHub repository to Render
2. Set environment variables:
   - `DATABASE_URL`: Your PostgreSQL connection string
   - `NETLIFY_FRONTEND_URL`: Your frontend URL (optional)
3. Render will automatically use the `Procfile` for deployment

### Environment Variables
- `DATABASE_URL`: PostgreSQL connection string (required)
- `NETLIFY_FRONTEND_URL`: Frontend URL for CORS (optional, defaults to localhost:3000)
- `FLASK_ENV`: Set to `production` for production deployments

## 🔧 Development

### Running Tests
```bash
# Add your test commands here
python -m pytest tests/
```

### Database Management
The application automatically initializes the database schema on startup. To reset the database:
```bash
# The engine drops and recreates tables on startup in development mode
python -c "from regicide_engine import initialize_database; initialize_database()"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Regicide card game designed by Badgers From Mars
- Special thanks to the Regicide community for rules clarification and support
- Built with Flask and the Python ecosystem

## 📞 Support

If you encounter any issues or have questions:
1. Check the [API documentation](./api_docs.md)
2. Review the [game rules](./regicide_rules.md)
3. Open an issue on GitHub
4. Contact the development team

---

**Happy Gaming! May you defeat all the royals! 👑⚔️**