"""
Semantic search interface for querying courses using word2vec.
"""

import json
import re
import os
from typing import List, Dict, Tuple, Optional
import numpy as np
from gensim.models import Word2Vec
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import MongoDB storage (optional dependency)
try:
    from .mongodb_storage import MongoDBStorage
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    logger.warning("MongoDB storage not available. Install pymongo to use MongoDB features.")


class SemanticSearch:
    def __init__(self, 
                 model_path: Optional[str] = 'word2vec_model.model', 
                 courses_path: str = 'courses.json',
                 use_mongodb: bool = False,
                 mongodb_uri: Optional[str] = None,
                 mongodb_db: str = 'course_search',
                 mongodb_collection: str = 'courses',
                 use_cosine_aggregation: bool = False):
        """
        Initialize semantic search with word2vec model and course data.
        
        Args:
            model_path: Path to trained word2vec model (optional if use_mongodb=True)
            courses_path: Path to courses JSON file
            use_mongodb: If True, use MongoDB for vector search
            mongodb_uri: MongoDB connection string (optional, uses env var)
            mongodb_db: MongoDB database name
            mongodb_collection: MongoDB collection name
            use_cosine_aggregation: If True, use MongoDB aggregation for cosine similarity
                                   instead of Atlas vector search (default: False)
        """
        self.use_mongodb = use_mongodb and MONGODB_AVAILABLE
        self.use_cosine_aggregation = use_cosine_aggregation
        self.model_path = model_path
        self.model = None  # Lazy-loaded when needed
        
        # Initialize MongoDB if requested
        self.mongodb_storage = None
        mongodb_connection_successful = False
        if self.use_mongodb:
            try:
                # Use provided values or fall back to environment variables
                db_name = mongodb_db or os.getenv('MONGODB_DATABASE', 'course_search')
                coll_name = mongodb_collection or os.getenv('MONGODB_COLLECTION', 'courses')
                
                self.mongodb_storage = MongoDBStorage(
                    connection_string=mongodb_uri,
                    database_name=db_name,
                    collection_name=coll_name
                )
                if self.mongodb_storage.connect():
                    mongodb_connection_successful = True
                    search_method = ("cosine similarity aggregation" if
                                     use_cosine_aggregation else
                                     "Atlas vector search")
                    logger.info(
                        f"Using MongoDB for vector search ({search_method})"
                    )
                    
                    # Check if word vectors are stored in MongoDB
                    if self.mongodb_storage.has_word_vectors():
                        logger.info(
                            "Word vectors found in MongoDB. Model file not "
                            "required for query vector generation."
                        )
                        # Model path is optional when word vectors are in MongoDB
                    elif model_path and os.path.exists(model_path):
                        logger.info(
                            "Word vectors not found in MongoDB. Model file "
                            "will be used for query vector generation. "
                            "Consider running --populate-mongodb to store "
                            "word vectors in MongoDB."
                        )
                    else:
                        logger.warning(
                            "Word vectors not found in MongoDB and model file "
                            "not found. Query vector generation will fail. "
                            "Either:\n"
                            "  1. Run --populate-mongodb to store word vectors, or\n"
                            "  2. Provide a valid model_path"
                        )
                else:
                    logger.warning("MongoDB connection failed, falling back to local model")
                    self.use_mongodb = False
            except Exception as e:
                logger.warning(f"MongoDB initialization failed: {e}, falling back to local model")
                self.use_mongodb = False
        
        # Load model if not using MongoDB (required for local search)
        if not self.use_mongodb:
            # Check if model is available
            if not model_path:
                if mongodb_connection_successful:
                    # MongoDB was requested and connected, but then disabled
                    raise ValueError(
                        "MongoDB connection was successful but then disabled. "
                        "This is unexpected. Please check your MongoDB "
                        "configuration."
                    )
                else:
                    raise ValueError(
                        "model_path is required when not using MongoDB. "
                        "Either provide a model_path or ensure MongoDB "
                        "connection succeeds."
                    )
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model file not found: {model_path}. "
                    "Please run with --train first, or ensure MongoDB "
                    "connection succeeds."
                )
            self.model = Word2Vec.load(model_path)
            logger.info(f"Loaded word2vec model from {model_path}")
        
        # Load courses (for local search or fallback)
        try:
            with open(courses_path, 'r', encoding='utf-8') as f:
                self.courses = json.load(f)
            logger.info(f"Loaded {len(self.courses)} courses from {courses_path}")
        except FileNotFoundError:
            logger.warning(f"Courses file not found: {courses_path}")
            self.courses = []
        
        # Precompute course vectors (only for local search)
        if not self.use_mongodb:
            self.course_vectors = {}
            self._precompute_course_vectors()
        else:
            self.course_vectors = {}
            logger.info("Skipping local vector precomputation (using MongoDB)")
    
    def _ensure_model_loaded(self):
        """Lazy-load the model when needed for query vector generation."""
        # If MongoDB has word vectors, we don't need the local model
        if (self.use_mongodb and self.mongodb_storage and
                self.mongodb_storage.has_word_vectors()):
            return
        
        if self.model is None:
            if not self.model_path:
                error_msg = (
                    "Model path required for query vector generation. "
                    "Either provide a model_path or ensure word vectors are "
                    "stored in MongoDB (run --populate-mongodb)."
                )
                raise ValueError(error_msg)
            if not os.path.exists(self.model_path):
                error_msg = (
                    f"Model file not found: {self.model_path}. "
                    "Required for generating query vectors. "
                    "Either provide a valid model file or ensure word vectors "
                    "are stored in MongoDB (run --populate-mongodb)."
                )
                raise FileNotFoundError(error_msg)
            self.model = Word2Vec.load(self.model_path)
            logger.info(f"Loaded word2vec model from {self.model_path}")
    
    def _get_text_tokens(self, text: str) -> List[str]:
        """Extract tokens from text (simple tokenization for query)."""
        if not text:
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep course codes
        text = re.sub(r'[^\w\s\-]', ' ', text)
        
        # Simple tokenization
        tokens = text.split()
        
        # If using MongoDB and word vectors are available, check vocabulary there
        if (self.use_mongodb and self.mongodb_storage and
                self.mongodb_storage.has_word_vectors()):
            # Filter tokens that exist in MongoDB vocabulary
            valid_tokens = []
            for token in tokens:
                if self.mongodb_storage.get_word_vector(token) is not None:
                    valid_tokens.append(token)
            return valid_tokens
        
        # Otherwise, use local model
        self._ensure_model_loaded()
        valid_tokens = [t for t in tokens if t in self.model.wv.key_to_index]
        
        return valid_tokens
    
    def _get_text_vector(self, text: str) -> np.ndarray:
        """
        Get average word vector for a text string.
        
        Args:
            text: Input text
            
        Returns:
            Average word vector (or zero vector if no valid tokens)
        """
        tokens = self._get_text_tokens(text)
        
        if not tokens:
            # Determine vector size
            vector_size = None
            
            # Try to get from MongoDB first
            if (self.use_mongodb and self.mongodb_storage and
                    self.mongodb_storage.has_word_vectors()):
                vector_size = self.mongodb_storage.get_vector_size()
            
            # Fallback to local model
            if vector_size is None:
                self._ensure_model_loaded()
                vector_size = self.model.wv.vector_size
            
            return np.zeros(vector_size)
        
        # Get vectors for each token
        vectors = []
        
        # Try MongoDB first if available
        if (self.use_mongodb and self.mongodb_storage and
                self.mongodb_storage.has_word_vectors()):
            word_vectors = self.mongodb_storage.get_word_vectors_batch(tokens)
            vectors = [word_vectors[token] for token in tokens
                      if token in word_vectors]
        else:
            # Fallback to local model
            self._ensure_model_loaded()
            vectors = [self.model.wv[token] for token in tokens]
        
        if not vectors:
            # Return zero vector if no valid vectors found
            vector_size = None
            if (self.use_mongodb and self.mongodb_storage and
                    self.mongodb_storage.has_word_vectors()):
                vector_size = self.mongodb_storage.get_vector_size()
            if vector_size is None:
                self._ensure_model_loaded()
                vector_size = self.model.wv.vector_size
            return np.zeros(vector_size)
        
        # Average the vectors
        avg_vector = np.mean(vectors, axis=0)
        
        return avg_vector
    
    def _precompute_course_vectors(self):
        """Precompute vectors for all courses."""
        logger.info("Precomputing course vectors...")
        
        for course in self.courses:
            # Combine all text fields
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
            vector = self._get_text_vector(course_text)
            
            # Store vector with course index
            course_id = course.get('code') or course.get('url', 'unknown')
            self.course_vectors[course_id] = {
                'vector': vector,
                'course': course
            }
        
        logger.info(f"Precomputed vectors for {len(self.course_vectors)} courses")
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (-1 to 1)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search for courses semantically similar to the query.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of (course_dict, similarity_score) tuples, sorted by similarity
        """
        # Get query vector
        query_vector = self._get_text_vector(query)
        
        if np.all(query_vector == 0):
            logger.warning(f"No valid tokens found in query: '{query}'")
            return []
        
        # Use MongoDB if enabled
        if self.use_mongodb and self.mongodb_storage:
            try:
                return self.mongodb_storage.vector_search(
                    query_vector, 
                    top_k=top_k,
                    use_cosine_aggregation=self.use_cosine_aggregation
                )
            except Exception as e:
                logger.error(f"MongoDB search failed: {e}, falling back to local")
                # Fall through to local search
        
        # Local search (fallback or default)
        if not self.course_vectors:
            # Recompute if needed
            self._precompute_course_vectors()
        
        # Calculate similarity with all courses
        similarities = []
        for course_id, course_data in self.course_vectors.items():
            similarity = self.cosine_similarity(
                query_vector, course_data['vector']
            )
            similarities.append((course_data['course'], similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k
        return similarities[:top_k]
    
    def find_similar_courses(self, course_code: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Find courses similar to a given course code.
        
        Args:
            course_code: Course code (e.g., "CPTR 141")
            top_k: Number of similar courses to return
            
        Returns:
            List of (course_dict, similarity_score) tuples
        """
        # Find the course (check MongoDB first if enabled)
        target_course = self.get_course_info(course_code)
        
        if not target_course:
            logger.warning(f"Course {course_code} not found")
            return []
        
        # Get course text
        text_parts = []
        if target_course.get('code'):
            text_parts.append(target_course['code'])
        if target_course.get('title'):
            text_parts.append(target_course['title'])
        if target_course.get('description'):
            text_parts.append(target_course['description'])
        
        query = ' '.join(text_parts)
        
        # Search for similar courses, excluding the target course itself
        results = self.search(query, top_k=top_k + 1)
        
        # Filter out the target course
        filtered_results = [
            (course, score) for course, score in results
            if course.get('code', '').upper() != course_code.upper()
        ]
        
        return filtered_results[:top_k]
    
    def get_course_info(self, course_code: str) -> Optional[Dict]:
        """
        Get detailed information about a specific course.
        
        Args:
            course_code: Course code (e.g., "CPTR 141")
            
        Returns:
            Course dictionary or None if not found
        """
        # Try MongoDB first if enabled
        if self.use_mongodb and self.mongodb_storage:
            try:
                course = self.mongodb_storage.get_course(course_code)
                if course:
                    return course
            except Exception as e:
                logger.warning(f"MongoDB lookup failed: {e}, using local")
        
        # Fallback to local search
        for course in self.courses:
            if course.get('code', '').upper() == course_code.upper():
                return course
        return None
    
    def toggle_mongodb(self, enable: bool, mongodb_uri: Optional[str] = None):
        """
        Toggle MongoDB usage on/off.
        
        Args:
            enable: If True, enable MongoDB; if False, use local model
            mongodb_uri: MongoDB connection string (optional)
        """
        if enable and not MONGODB_AVAILABLE:
            logger.error("MongoDB not available. Install pymongo to use MongoDB.")
            return False
        
        if enable:
            try:
                # Use provided URI or fall back to environment variable
                uri = mongodb_uri or os.getenv('MONGODB_URI')
                db_name = os.getenv('MONGODB_DATABASE', 'course_search')
                coll_name = os.getenv('MONGODB_COLLECTION', 'courses')
                
                self.mongodb_storage = MongoDBStorage(
                    connection_string=uri,
                    database_name=db_name,
                    collection_name=coll_name
                )
                if self.mongodb_storage.connect():
                    self.use_mongodb = True
                    search_method = "cosine similarity aggregation" if self.use_cosine_aggregation else "Atlas vector search"
                    logger.info(f"Switched to MongoDB vector search ({search_method})")
                    return True
                else:
                    logger.error("Failed to connect to MongoDB")
                    return False
            except Exception as e:
                logger.error(f"Failed to enable MongoDB: {e}")
                return False
        else:
            self.use_mongodb = False
            if self.mongodb_storage:
                self.mongodb_storage.disconnect()
            # Ensure model is loaded for local search
            try:
                self._ensure_model_loaded()
            except (ValueError, FileNotFoundError) as e:
                logger.error(f"Cannot switch to local model: {e}")
                return False
            # Recompute local vectors if needed
            if not self.course_vectors:
                self._precompute_course_vectors()
            logger.info("Switched to local model")
            return True
    
    def toggle_cosine_aggregation(self, enable: bool) -> bool:
        """
        Toggle between Atlas vector search and MongoDB cosine similarity aggregation.
        
        Args:
            enable: If True, use cosine similarity aggregation; if False, use Atlas vector search
            
        Returns:
            True if toggled successfully, False otherwise
        """
        if not self.use_mongodb:
            logger.warning("MongoDB must be enabled to use cosine similarity aggregation")
            return False
        
        self.use_cosine_aggregation = enable
        method = "cosine similarity aggregation" if enable else "Atlas vector search"
        logger.info(f"Switched to {method}")
        return True
    
    def _extract_course_from_query(self, query: str) -> Dict:
        """
        Extract course information from a natural language query.
        Uses semantic search to find the most relevant course.
        
        Args:
            query: Natural language query mentioning a course
            
        Returns:
            Course dictionary or None if not found
        """
        # First, try to find exact course code match (e.g., "CPTR 354")
        import re
        code_pattern = r'\b(CPTR|CIS|CYBS|GDEV)\s*[-\s]?\s*(\d{3})\b'
        code_match = re.search(code_pattern, query, re.IGNORECASE)
        if code_match:
            code = f"{code_match.group(1).upper()} {code_match.group(2)}"
            course = self.get_course_info(code)
            if course:
                return course
        
        # Use semantic search to find the course
        results = self.search(query, top_k=5)
        if results:
            # Return the most similar course
            return results[0][0]
        
        return None
    
    def answer_question(self, question: str) -> str:
        """
        Answer a natural language question about courses.
        
        Supported question types:
        - Prerequisites: "What classes do I need before taking X?"
        - Corequisites: "What classes do I take with X?"
        - Description: "What is X about?" or "Tell me about X"
        - Credits: "How many credits is X?"
        
        Args:
            question: Natural language question
            
        Returns:
            Natural language answer
        """
        question_lower = question.lower()
        
        # Extract course from question
        course = self._extract_course_from_query(question)
        
        if not course:
            return (
                "I couldn't find that course. "
                "Please try rephrasing your question or include the course code."
            )
        
        course_code = course.get('code', 'Unknown')
        course_title = course.get('title', '')
        
        # Determine question type and generate answer
        if any(keyword in question_lower for keyword in [
            'prerequisite', 'before', 'need', 'required', 'take before'
        ]):
            # Prerequisites question
            prereqs = course.get('prerequisites', '')
            if prereqs:
                return (
                    f"Before taking {course_code} - {course_title}, "
                    f"you need: {prereqs}."
                )
            else:
                return (
                    f"{course_code} - {course_title} has no prerequisites "
                    f"listed (or may require permission of instructor)."
                )
        
        elif any(keyword in question_lower for keyword in [
            'corequisite', 'take with', 'together', 'concurrent'
        ]):
            # Corequisites question
            coreqs = course.get('corequisites', '')
            if coreqs:
                return (
                    f"You can take {course_code} - {course_title} "
                    f"together with: {coreqs}."
                )
            else:
                return (
                    f"{course_code} - {course_title} has no corequisites listed."
                )
        
        elif any(keyword in question_lower for keyword in [
            'credit', 'credits', 'hours'
        ]):
            # Credits question
            credits = course.get('credits', '')
            if credits:
                return (
                    f"{course_code} - {course_title} is worth {credits} credits."
                )
            else:
                return (
                    f"I don't have credit information for "
                    f"{course_code} - {course_title}."
                )
        
        elif any(keyword in question_lower for keyword in [
            'about', 'what is', 'describe', 'tell me', 'explain'
        ]):
            # Description question
            description = course.get('description', '')
            if description:
                return (
                    f"{course_code} - {course_title}\n\n"
                    f"{description}"
                )
            else:
                return (
                    f"I don't have a description for "
                    f"{course_code} - {course_title}."
                )
        
        else:
            # Default: return general course information
            info_parts = [f"{course_code} - {course_title}"]
            
            if course.get('description'):
                desc = course['description']
                if len(desc) > 200:
                    desc = desc[:200] + '...'
                info_parts.append(f"\nDescription: {desc}")
            
            if course.get('prerequisites'):
                info_parts.append(
                    f"\nPrerequisites: {course['prerequisites']}"
                )
            
            if course.get('credits'):
                info_parts.append(f"\nCredits: {course['credits']}")
            
            return '\n'.join(info_parts)


if __name__ == '__main__':
    # Example usage
    search = SemanticSearch()
    
    # Example queries
    queries = [
        "machine learning",
        "web development",
        "databases",
        "operating systems",
        "programming fundamentals"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: '{query}'")
        print('='*60)
        results = search.search(query, top_k=3)
        for i, (course, score) in enumerate(results, 1):
            print(f"\n{i}. {course.get('code', 'N/A')} - {course.get('title', 'N/A')}")
            print(f"   Similarity: {score:.4f}")
            if course.get('description'):
                desc = course['description'][:150] + '...' if len(course['description']) > 150 else course['description']
                print(f"   {desc}")

