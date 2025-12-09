"""
Semantic search interface for querying courses using word2vec.
"""

import json
import re
from typing import List, Dict, Tuple
import numpy as np
from gensim.models import Word2Vec
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticSearch:
    def __init__(self, model_path: str = 'word2vec_model.model', 
                 courses_path: str = 'courses.json'):
        """
        Initialize semantic search with word2vec model and course data.
        
        Args:
            model_path: Path to trained word2vec model
            courses_path: Path to courses JSON file
        """
        # Load model
        self.model = Word2Vec.load(model_path)
        logger.info(f"Loaded word2vec model from {model_path}")
        
        # Load courses
        with open(courses_path, 'r', encoding='utf-8') as f:
            self.courses = json.load(f)
        logger.info(f"Loaded {len(self.courses)} courses")
        
        # Precompute course vectors
        self.course_vectors = {}
        self._precompute_course_vectors()
    
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
        
        # Filter tokens that exist in model vocabulary
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
            # Return zero vector if no valid tokens
            return np.zeros(self.model.wv.vector_size)
        
        # Get vectors for each token
        vectors = [self.model.wv[token] for token in tokens]
        
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
        
        # Calculate similarity with all courses
        similarities = []
        for course_id, course_data in self.course_vectors.items():
            similarity = self.cosine_similarity(query_vector, course_data['vector'])
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
        # Find the course
        target_course = None
        for course in self.courses:
            if course.get('code', '').upper() == course_code.upper():
                target_course = course
                break
        
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
    
    def get_course_info(self, course_code: str) -> Dict:
        """
        Get detailed information about a specific course.
        
        Args:
            course_code: Course code (e.g., "CPTR 141")
            
        Returns:
            Course dictionary or None if not found
        """
        for course in self.courses:
            if course.get('code', '').upper() == course_code.upper():
                return course
        return None
    
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

