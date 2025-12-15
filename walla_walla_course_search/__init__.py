"""
Walla Walla Course Search - Semantic search for Walla Walla University Computer Science courses.
"""

from .scraper import CourseScraper
from .word2vec_trainer import Word2VecTrainer
from .semantic_search import SemanticSearch

try:
    from .mongodb_storage import MongoDBStorage
    __all__ = ['CourseScraper', 'Word2VecTrainer', 'SemanticSearch', 'MongoDBStorage']
except ImportError:
    __all__ = ['CourseScraper', 'Word2VecTrainer', 'SemanticSearch']

