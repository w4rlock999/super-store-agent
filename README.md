# Super Store Agent

A fullstack application combining Next.js frontend with a Python Flask backend for an AI-powered Shopify Admin agent.

## Features

- Next.js 15 frontend with TypeScript
- Python Flask backend with AI agent capabilities
- Real-time chat interface
- Travel itinerary planning with web search capabilities

## Setup

1. Install frontend dependencies:
```bash
npm install
```

2. Set up Python virtual environment:
```bash
cd server
python -m venv .venv

# On Windows
.venv\Scripts\activate

# On macOS/Linux
source .venv/bin/activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Add your OpenAI API key to .env
```

5. Start the development servers:

Frontend (Next.js):
```bash
npm run dev
```

Backend (Flask):
```bash
python server/server.py
```

## Project Structure

- `app/` - Next.js frontend application
- `server/` - Python Flask backend with AI agents
- `public/` - Static assets

## Development

- Frontend runs on `http://localhost:3000`
- Backend runs on `http://localhost:5000`

