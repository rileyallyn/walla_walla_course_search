"""
Train word2vec model on scraped course data for semantic search.
"""

import json
import re
from typing import List, Dict
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Download required NLTK data
def _download_nltk_data():
    """Download required NLTK data resources."""
    # Download punkt_tab (newer NLTK versions) - this is what's needed
    punkt_tab_available = False
    punkt_available = False
    
    try:
        nltk.data.find('tokenizers/punkt_tab')
        punkt_tab_available = True
    except LookupError:
        pass
    
    try:
        nltk.data.find('tokenizers/punkt')
        punkt_available = True
    except LookupError:
        pass
    
    # Download punkt_tab if not available (required for newer NLTK)
    if not punkt_tab_available:
        try:
            logger.info("Downloading punkt_tab tokenizer...")
            nltk.download('punkt_tab', quiet=True)
            # Verify it was downloaded
            try:
                nltk.data.find('tokenizers/punkt_tab')
                punkt_tab_available = True
                logger.info("punkt_tab downloaded successfully")
            except LookupError:
                logger.warning("punkt_tab download may have failed")
        except Exception as e:
            logger.warning(f"Failed to download punkt_tab: {e}")
    
    # Also download punkt as fallback (for older NLTK or compatibility)
    if not punkt_available:
        try:
            logger.info("Downloading punkt tokenizer...")
            nltk.download('punkt', quiet=True)
            # Verify it was downloaded
            try:
                nltk.data.find('tokenizers/punkt')
                punkt_available = True
                logger.info("punkt downloaded successfully")
            except LookupError:
                logger.warning("punkt download may have failed")
        except Exception as e:
            logger.warning(f"Failed to download punkt: {e}")

    # Download stopwords
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        logger.info("Downloading stopwords...")
        nltk.download('stopwords', quiet=True)

    # Download wordnet
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        logger.info("Downloading wordnet...")
        nltk.download('wordnet', quiet=True)

# Download NLTK data on import
_download_nltk_data()



class EpochLogger(CallbackAny2Vec):
    """Callback to log information about training."""
    
    def __init__(self):
        self.epoch = 0
    
    def on_epoch_end(self, model):
        logger.info(f"Epoch #{self.epoch} end")
        self.epoch += 1


class Word2VecTrainer:
    def __init__(self, courses_data: List[Dict]):
        """
        Initialize the trainer with course data.
        
        Args:
            courses_data: List of course dictionaries from scraper
        """
        self.courses = courses_data
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            logger.warning("NLTK stopwords not available, continuing without stopword removal")
            self.stop_words = set()
        
        try:
            self.lemmatizer = WordNetLemmatizer()
        except LookupError:
            logger.warning("NLTK wordnet not available, continuing without lemmatization")
            self.lemmatizer = None
        
        self.model = None
        self.processed_sentences = []
        
    def _tokenize_words(self, text: str) -> List[str]:
        """
        Tokenize text into words with fallback if NLTK fails.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of word tokens
        """
        # Check if punkt_tab is available, download if not
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                # Neither is available, try to download punkt_tab
                try:
                    logger.info("Downloading punkt_tab for word tokenization...")
                    nltk.download('punkt_tab', quiet=True)
                except Exception as e:
                    logger.warning(f"Could not download punkt_tab: {e}")
        
        try:
            return word_tokenize(text)
        except LookupError:
            # Fallback: simple word splitting on whitespace and punctuation
            logger.warning(
                "NLTK word tokenizer not available, using simple fallback"
            )
            # Split on whitespace and keep alphanumeric sequences
            tokens = re.findall(r'\b\w+\b', text)
            return tokens
    
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text: tokenize, lowercase, remove stopwords, lemmatize.
        
        Args:
            text: Input text string
            
        Returns:
            List of preprocessed tokens
        """
        if not text:
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep course codes and numbers
        # Preserve patterns like "CPTR 141" or "cptr-141"
        text = re.sub(r'[^\w\s\-]', ' ', text)
        
        # Tokenize with fallback
        tokens = self._tokenize_words(text)
        
        # Filter and process tokens
        processed = []
        for token in tokens:
            # Skip very short tokens (except course numbers)
            if len(token) < 2 and not token.isdigit():
                continue
            
            # Skip stopwords (but keep course codes)
            if token in self.stop_words and not any(char.isupper() for char in token):
                continue
            
            # Lemmatize (convert to base form) if available
            if self.lemmatizer:
                lemma = self.lemmatizer.lemmatize(token)
            else:
                lemma = token
            
            # Keep the token
            processed.append(lemma)
        
        return processed
    
    def _extract_course_text(self, course: Dict) -> str:
        """
        Extract and combine all text fields from a course.
        
        Args:
            course: Course dictionary
            
        Returns:
            Combined text string
        """
        text_parts = []
        
        # Add course code
        if course.get('code'):
            text_parts.append(course['code'])
        
        # Add title
        if course.get('title'):
            text_parts.append(course['title'])
        
        # Add description
        if course.get('description'):
            text_parts.append(course['description'])
        
        # Add prerequisites if available
        if course.get('prerequisites'):
            prereq_text = f"Prerequisites: {course['prerequisites']}"
            text_parts.append(prereq_text)
        
        # Add full text if description is missing
        if not course.get('description') and course.get('full_text'):
            text_parts.append(course['full_text'])
        
        return ' '.join(text_parts)
    
    def _tokenize_sentences(self, text: str) -> List[str]:
        """
        Tokenize text into sentences with fallback if NLTK fails.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of sentences
        """
        # Check if punkt_tab is available, download if not
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                # Neither is available, try to download punkt_tab
                try:
                    logger.info("Downloading punkt_tab for sentence tokenization...")
                    nltk.download('punkt_tab', quiet=True)
                except Exception as e:
                    logger.warning(f"Could not download punkt_tab: {e}")
        
        try:
            return sent_tokenize(text)
        except LookupError:
            # Fallback: simple sentence splitting on periods,
            # exclamation, question marks
            # This is a basic fallback if NLTK resources aren't available
            logger.warning(
                "NLTK sentence tokenizer not available, "
                "using simple fallback"
            )
            import re
            # Split on sentence-ending punctuation,
            # but preserve the punctuation
            sentences = re.split(r'([.!?]+)', text)
            # Recombine sentences with their punctuation
            result = []
            for i in range(0, len(sentences) - 1, 2):
                sentence = sentences[i].strip()
                if i + 1 < len(sentences):
                    sentence += sentences[i + 1]
                if sentence:
                    result.append(sentence)
            # Add last sentence if it exists and wasn't paired
            if len(sentences) % 2 == 1 and sentences[-1].strip():
                result.append(sentences[-1].strip())
            return result if result else [text]
    
    def prepare_training_data(self):
        """Prepare sentences from course data for training."""
        logger.info("Preparing training data...")
        
        for course in self.courses:
            course_text = self._extract_course_text(course)
            
            if not course_text:
                continue
            
            # Split into sentences with fallback
            sentences = self._tokenize_sentences(course_text)
            
            # Process each sentence
            for sentence in sentences:
                processed = self._preprocess_text(sentence)
                if len(processed) >= 2:  # Need at least 2 words for context
                    self.processed_sentences.append(processed)
        
        logger.info(f"Prepared {len(self.processed_sentences)} sentences for training")
        
        # Log some statistics
        if self.processed_sentences:
            avg_len = sum(len(s) for s in self.processed_sentences) / len(self.processed_sentences)
            logger.info(f"Average sentence length: {avg_len:.2f} tokens")
    
    def train(self, 
              vector_size: int = 100,
              window: int = 5,
              min_count: int = 2,
              workers: int = 4,
              epochs: int = 100,
              sg: int = 1):
        """
        Train the word2vec model.
        
        Args:
            vector_size: Dimensionality of word vectors
            window: Maximum distance between current and predicted word
            min_count: Minimum count of words to include in vocabulary
            workers: Number of worker threads
            epochs: Number of training epochs
            sg: Training algorithm: 1 for skip-gram, 0 for CBOW
        """
        if not self.processed_sentences:
            raise ValueError("No training data. Call prepare_training_data() first.")
        
        logger.info("Training word2vec model...")
        logger.info(f"Parameters: vector_size={vector_size}, window={window}, "
                   f"min_count={min_count}, epochs={epochs}, sg={sg}")
        
        epoch_logger = EpochLogger()
        
        self.model = Word2Vec(
            sentences=self.processed_sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=sg,
            epochs=epochs,
            callbacks=[epoch_logger]
        )
        
        logger.info("Training complete!")
        logger.info(f"Vocabulary size: {len(self.model.wv.key_to_index)}")
    
    def save_model(self, filepath: str = 'word2vec_model.model'):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = 'word2vec_model.model'):
        """Load a trained model from disk."""
        self.model = Word2Vec.load(filepath)
        logger.info(f"Model loaded from {filepath}")
    
    def get_model(self):
        """Get the trained model."""
        return self.model


if __name__ == '__main__':
    # Load courses data
    if not os.path.exists('courses.json'):
        logger.error("courses.json not found. Please run scraper.py first.")
        exit(1)
    
    with open('courses.json', 'r', encoding='utf-8') as f:
        courses = json.load(f)
    
    logger.info(f"Loaded {len(courses)} courses")
    
    # Train model
    trainer = Word2VecTrainer(courses)
    trainer.prepare_training_data()
    trainer.train(epochs=100, vector_size=100)
    trainer.save_model()

