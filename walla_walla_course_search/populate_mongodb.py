"""
Script to populate MongoDB with course embeddings.
"""

import json
import os
import sys
import argparse
import logging
import re
from typing import List, Dict
import numpy as np
from gensim.models import Word2Vec
from dotenv import load_dotenv
from .mongodb_storage import MongoDBStorage

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_embeddings(courses: List[Dict], model: Word2Vec) -> List[np.ndarray]:
    """
    Generate embeddings for all courses using the word2vec model.
    
    Args:
        courses: List of course dictionaries
        model: Trained Word2Vec model
        
    Returns:
        List of embedding vectors
    """
    embeddings = []
    
    for course in courses:
        # Combine all text fields (same as in SemanticSearch)
        text_parts = []
        if course.get('code'):
            text_parts.append(course['code'])
        if course.get('title'):
            text_parts.append(course['title'])
        if course.get('description'):
            text_parts.append(course['description'])
        if course.get('prerequisites'):
            text_parts.append(f"Prerequisites: {course['prerequisites']}")
        if course.get('full_text'):
            text_parts.append(course['full_text'])
        
        course_text = ' '.join(text_parts)
        
        # Get text vector (same logic as SemanticSearch._get_text_vector)
        text = course_text.lower()
        text = re.sub(r'[^\w\s\-]', ' ', text)
        tokens = text.split()
        valid_tokens = [
            t for t in tokens 
            if t in model.wv.key_to_index
        ]
        
        if not valid_tokens:
            # Return zero vector if no valid tokens
            embedding = np.zeros(model.wv.vector_size)
        else:
            # Get vectors for each token and average
            vectors = [model.wv[token] for token in valid_tokens]
            embedding = np.mean(vectors, axis=0)
        
        embeddings.append(embedding)
    
    return embeddings


def populate_mongodb(
    model_path: str,
    courses_path: str,
    mongodb_uri: str = None,
    database_name: str = 'course_search',
    collection_name: str = 'courses',
    clear_existing: bool = False
):
    """
    Populate MongoDB with course embeddings.
    
    Args:
        model_path: Path to word2vec model
        courses_path: Path to courses JSON file
        mongodb_uri: MongoDB connection string
        database_name: MongoDB database name
        collection_name: MongoDB collection name
        clear_existing: If True, clear collection before populating
    """
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = Word2Vec.load(model_path)
    
    # Load courses
    logger.info(f"Loading courses from {courses_path}")
    with open(courses_path, 'r', encoding='utf-8') as f:
        courses = json.load(f)
    
    logger.info(f"Loaded {len(courses)} courses")
    
    # Connect to MongoDB
    logger.info("Connecting to MongoDB...")
    # Use provided values or fall back to environment variables
    final_uri = mongodb_uri or os.getenv('MONGODB_URI')
    final_db = database_name or os.getenv('MONGODB_DATABASE', 'course_search')
    final_coll = collection_name or os.getenv('MONGODB_COLLECTION', 'courses')
    
    storage = MongoDBStorage(
        connection_string=final_uri,
        database_name=final_db,
        collection_name=final_coll
    )
    
    if not storage.connect():
        logger.error("Failed to connect to MongoDB")
        sys.exit(1)
    
    # Clear existing data if requested
    if clear_existing:
        logger.info("Clearing existing collection...")
        storage.clear_collection()
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    embeddings = generate_embeddings(courses, model)
    logger.info(f"Generated {len(embeddings)} embeddings")
    
    # Store in MongoDB
    logger.info("Storing embeddings in MongoDB...")
    stored_count = storage.store_courses_batch(
        courses, embeddings, batch_size=50
    )
    
    logger.info(f"Successfully stored {stored_count} courses")
    
    # Store Word2Vec vocabulary and word vectors in MongoDB
    logger.info("Storing Word2Vec vocabulary in MongoDB...")
    if storage.store_word_vectors(model):
        logger.info("Word vectors stored successfully")
    else:
        logger.warning("Failed to store word vectors")
    
    # Create vector index
    logger.info("Creating vector search index...")
    try:
        # Get vector size from model
        vector_dimensions = model.wv.vector_size
        storage.create_vector_index(vector_dimensions=vector_dimensions)
        logger.info("Vector index creation process completed")
    except Exception as e:
        logger.warning(f"Could not create vector index: {e}")
        logger.info("You may need to create the index manually in Atlas")
    
    # Verify
    count = storage.count_courses()
    logger.info(f"Collection now contains {count} courses")
    
    storage.disconnect()
    logger.info("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Populate MongoDB with course embeddings'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='word2vec_model.model',
        help='Path to word2vec model'
    )
    parser.add_argument(
        '--courses',
        type=str,
        default='courses.json',
        help='Path to courses JSON file'
    )
    parser.add_argument(
        '--mongodb-uri',
        type=str,
        default=None,
        help='MongoDB connection string (or set MONGODB_URI env var)'
    )
    parser.add_argument(
        '--database',
        type=str,
        default='course_search',
        help='MongoDB database name'
    )
    parser.add_argument(
        '--collection',
        type=str,
        default='courses',
        help='MongoDB collection name'
    )
    parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear existing collection before populating'
    )
    
    args = parser.parse_args()
    
    populate_mongodb(
        model_path=args.model,
        courses_path=args.courses,
        mongodb_uri=args.mongodb_uri,
        database_name=args.database,
        collection_name=args.collection,
        clear_existing=args.clear
    )
