
# Smart Research Agent

A Python console-based AI agent that automatically selects tools to answer user queries.

## Features
- Web search using DuckDuckGo
- Python arithmetic calculator
- Local TXT file reader
- Groq LLM function calling
- Tool actions, observations, and final answers
- Supports compound research queries

## Technologies
- Python
- Groq API
- DuckDuckGo Search (`ddgs`)
- python-dotenv

## Setup Instructions

1. Clone or download this repository.
2. Create a virtual environment:

   `python -m venv .venv`

3. Activate it on Windows:

   `.venv\Scripts\activate`

4. Install dependencies:

   `pip install -r requirements.txt`

5. Create a `.env` file and add your Groq API key:

   `GROQ_API_KEY=your_api_key_here`

6. Run the application:

   `python main.py`

## Sample Queries
- Calculate 15% of 250000.
- Read sample.txt and summarize it.
- Search the web for the latest Python release.

## Security
Do not upload API keys or the `.env` file to GitHub.
