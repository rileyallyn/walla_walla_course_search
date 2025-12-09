# Walla Walla University Computer Science Course Semantic Search

This project uses word2vec to enable semantic search over the Walla Walla University Computer Science course catalog. It scrapes course information from the catalog website and trains a word2vec model to find courses semantically similar to natural language queries.

## Features

- **Recursive Web Scraping**: Automatically crawls the course catalog website to extract course information
- **Word2Vec Training**: Trains a word2vec model on course descriptions and content
- **Semantic Search**: Query courses using natural language (e.g., "machine learning", "web development")
- **Interactive Interface**: Command-line interface for exploring courses

## Installation

1. Install dependencies using `uv`:
```bash
uv sync
# or if you prefer using requirements.txt:
uv pip install -r requirements.txt
```

2. Download required NLTK data (done automatically on first run, but can be done manually):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Usage

### Complete Pipeline (Scrape → Train → Search)

Run everything in one command:
```bash
uv run python main.py --all
```

### Step-by-Step

1. **Scrape courses from the website:**
```bash
uv run python main.py --scrape
```
This creates `courses.json` with all scraped course information.

2. **Train the word2vec model:**
```bash
uv run python main.py --train
```
This trains the model and saves it as `word2vec_model.model`.

3. **Interactive search:**
```bash
uv run python main.py --search
```

4. **Single query (non-interactive):**
```bash
uv run python main.py --query "machine learning"
```

### Examples

```bash
# Scrape only
uv run python main.py --scrape

# Train with custom parameters
uv run python main.py --train --epochs 200 --vector-size 150

# Search interactively
uv run python main.py --search

# Run a single query
uv run python main.py --query "web development"

# Find courses similar to a specific course
uv run python main.py --search
# Then type: "similar to CPTR 141"
```

**Note:** You can also use `python` directly if you've activated the virtual environment that `uv` created:
```bash
source .venv/bin/activate  # On macOS/Linux
python main.py --all
```

## Project Structure

- `scraper.py`: Recursive web scraper for course catalog
- `word2vec_trainer.py`: Word2Vec model training on course data
- `semantic_search.py`: Semantic search interface using trained model
- `main.py`: Main orchestration script with CLI
- `pyproject.toml`: Project configuration and dependencies (for uv)
- `requirements.txt`: Python dependencies (alternative)
- `courses.json`: Scraped course data (generated)
- `word2vec_model.model`: Trained word2vec model (generated)

## How It Works

1. **Scraping**: The scraper recursively follows links from the main course catalog page, extracting course codes, titles, descriptions, and other metadata.

2. **Training**: Course text is preprocessed (tokenized, lemmatized, stopwords removed) and used to train a Word2Vec model that learns semantic relationships between words.

3. **Search**: User queries are converted to word vectors, and cosine similarity is used to find the most semantically similar courses.

## Query Examples

- "machine learning"
- "databases"
- "web development"
- "operating systems"
- "programming fundamentals"
- "artificial intelligence"
- "similar to CPTR 141"

## Notes

- The scraper includes a delay between requests to be respectful to the server
- The word2vec model is trained using skip-gram architecture by default
- Course vectors are precomputed for fast search performance
- The model vocabulary size depends on the diversity of course content

