"""
MongoDB storage for course embeddings and vector search.
"""

import json
import os
from typing import List, Dict, Optional, Tuple
import numpy as np
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, OperationFailure
from pymongo.operations import ReplaceOne
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MongoDBStorage:
    """Handle MongoDB operations for course embeddings."""
    
    def __init__(self, 
                 connection_string: Optional[str] = None,
                 database_name: Optional[str] = None,
                 collection_name: Optional[str] = None):
        """
        Initialize MongoDB connection.
        
        Args:
            connection_string: MongoDB connection string.
                              If None, reads from MONGODB_URI env var
            database_name: Name of the database (defaults to env var or 'course_search')
            collection_name: Name of the collection (defaults to env var or 'courses')
        """
        self.connection_string = (
            connection_string or 
            os.getenv('MONGODB_URI') or 
            os.getenv('MONGODB_CONNECTION_STRING')
        )
        
        self.database_name = (
            database_name or 
            os.getenv('MONGODB_DATABASE') or 
            'course_search'
        )
        
        self.collection_name = (
            collection_name or 
            os.getenv('MONGODB_COLLECTION') or 
            'courses'
        )
        
        if not self.connection_string:
            raise ValueError(
                "MongoDB connection string required. "
                "Set MONGODB_URI in .env file or pass connection_string"
            )
        
        self.client: Optional[MongoClient] = None
        self.collection: Optional[Collection] = None
        self.vocab_collection: Optional[Collection] = None
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to MongoDB."""
        try:
            self.client = MongoClient(
                self.connection_string,
                serverSelectionTimeoutMS=5000
            )
            # Test connection
            self.client.admin.command('ping')
            self.connected = True
            
            # Get database and collections
            db = self.client[self.database_name]
            self.collection = db[self.collection_name]
            # Vocabulary collection for storing word vectors
            self.vocab_collection = db[f"{self.collection_name}_vocab"]
            
            logger.info(
                f"Connected to MongoDB: {self.database_name}.{self.collection_name}"
            )
            return True
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self.connected = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            self.connected = False
            logger.info("Disconnected from MongoDB")
    
    def create_vector_index(self, vector_field: str = 'embedding', vector_dimensions: int = 100):
        """
        Create a vector search index on the embedding field.
        
        Args:
            vector_field: Name of the field containing the vector
            vector_dimensions: Number of dimensions in the vector (default: 100)
        """
        if not self.connected:
            raise ConnectionError("Not connected to MongoDB")
        
        try:
            # Check if search index already exists
            try:
                existing_indexes = list(self.collection.list_search_indexes())
                for idx in existing_indexes:
                    if idx.get('name') == 'vector_index' or 'vector' in idx.get('name', '').lower():
                        logger.info("Vector search index already exists")
                        return
            except Exception:
                # list_search_indexes might not be available
                pass
            
            # Try to create Atlas Vector Search index using create_search_index
            try:
                index_definition = {
                    "definition": {
                        "fields": [
                            {
                                "type": "vector",
                                "numDimensions": vector_dimensions,
                                "path": vector_field,
                                "similarity": "cosine"
                            }
                        ]
                    },
                    "name": "vector_index",
                    "type": "vectorSearch"
                }
                
                # Create the search index (this is async in Atlas)
                result = self.collection.create_search_index(index_definition)
                logger.info(
                    f"Vector search index creation initiated. "
                    f"Index ID: {result}. "
                    f"Note: Index creation is asynchronous in Atlas."
                )
                logger.info(
                    "You can check index status in MongoDB Atlas UI or "
                    "using list_search_indexes()"
                )
                return
            except AttributeError:
                # create_search_index method not available (older pymongo version)
                logger.warning(
                    "create_search_index method not available. "
                    "Please create the vector search index manually in MongoDB Atlas UI."
                )
                logger.info(
                    "To create manually:\n"
                    "1. Go to MongoDB Atlas → Search → Create Search Index\n"
                    "2. Use JSON Editor\n"
                    "3. Use this definition:\n"
                    f'  {{"fields": [{{"type": "vector", "numDimensions": {vector_dimensions}, '
                    f'"path": "{vector_field}", "similarity": "cosine"}}]}}'
                )
            except OperationFailure as e:
                # Index creation failed (might not have permissions or Atlas Search not enabled)
                logger.warning(
                    f"Vector search index creation failed: {e}. "
                    "This might require manual creation in Atlas UI."
                )
                logger.info(
                    "Vector search requires MongoDB Atlas with Atlas Search enabled. "
                    "The system will use manual cosine similarity calculation instead."
                )
            
            # Fallback: create regular index for basic queries
            try:
                self.collection.create_index([(vector_field, 1)])
                logger.info("Created regular index on embedding field as fallback")
            except Exception as e:
                logger.warning(f"Could not create regular index: {e}")
                
        except Exception as e:
            logger.error(f"Error creating vector index: {e}")
            # Don't raise - allow the system to continue with manual similarity
            logger.info("System will use manual cosine similarity calculation")
    
    def store_course_embedding(self, 
                              course: Dict,
                              embedding: np.ndarray,
                              vector_field: str = 'embedding') -> bool:
        """
        Store a course with its embedding in MongoDB.
        
        Args:
            course: Course dictionary
            embedding: Vector embedding as numpy array
            vector_field: Name of the field to store the embedding
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            raise ConnectionError("Not connected to MongoDB")
        
        try:
            # Convert numpy array to list for JSON serialization
            embedding_list = embedding.tolist()
            
            # Prepare document (exclude full_text to reduce size)
            document = course.copy()
            document.pop('full_text', None)  # Remove full_text field
            document[vector_field] = embedding_list
            document['_id'] = course.get('code') or course.get('url', '')
            
            # Upsert (insert or update)
            self.collection.replace_one(
                {'_id': document['_id']},
                document,
                upsert=True
            )
            
            return True
        except Exception as e:
            logger.error(f"Error storing course embedding: {e}")
            return False
    
    def store_courses_batch(self,
                            courses: List[Dict],
                            embeddings: List[np.ndarray],
                            vector_field: str = 'embedding',
                            batch_size: int = 100) -> int:
        """
        Store multiple courses with embeddings in batch.
        
        Args:
            courses: List of course dictionaries
            embeddings: List of embedding arrays
            vector_field: Name of the field to store the embedding
            batch_size: Number of documents to insert per batch
            
        Returns:
            Number of courses successfully stored
        """
        if not self.connected:
            raise ConnectionError("Not connected to MongoDB")
        
        if len(courses) != len(embeddings):
            raise ValueError(
                "Number of courses must match number of embeddings"
            )
        
        stored_count = 0
        
        for i in range(0, len(courses), batch_size):
            batch_courses = courses[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            
            documents = []
            for course, embedding in zip(batch_courses, batch_embeddings):
                embedding_list = embedding.tolist()
                document = course.copy()
                # Exclude full_text to reduce document size
                document.pop('full_text', None)
                document[vector_field] = embedding_list
                document['_id'] = course.get('code') or course.get('url', '')
                documents.append(document)
            
            try:
                # Use bulk_write for better performance
                operations = []
                for doc in documents:
                    doc_id = doc.get('_id')
                    operations.append(
                        ReplaceOne(
                            {'_id': doc_id},
                            doc,  # Document includes _id
                            upsert=True
                        )
                    )
                
                result = self.collection.bulk_write(operations)
                stored_count += result.upserted_count + result.modified_count
                logger.info(
                    f"Stored batch {i//batch_size + 1}: "
                    f"{len(batch_courses)} courses"
                )
            except Exception as e:
                logger.error(f"Error storing batch: {e}")
                # Fallback to individual inserts
                logger.info("Falling back to individual inserts...")
                for doc in documents:
                    try:
                        doc_id = doc.get('_id', '')
                        self.collection.replace_one(
                            {'_id': doc_id},
                            doc,
                            upsert=True
                        )
                        stored_count += 1
                    except Exception as e2:
                        logger.error(f"Error storing individual document: {e2}")
        
        logger.info(f"Successfully stored {stored_count} courses")
        return stored_count
    
    def vector_search(self,
                     query_vector: np.ndarray,
                     top_k: int = 5,
                     vector_field: str = 'embedding',
                     limit: int = 100,
                     use_cosine_aggregation: bool = False) -> List[Tuple[Dict, float]]:
        """
        Perform vector similarity search in MongoDB.
        
        Args:
            query_vector: Query vector as numpy array
            top_k: Number of top results to return
            vector_field: Name of the field containing the embedding
            limit: Maximum number of documents to consider (for Atlas vector search)
            use_cosine_aggregation: If True, use MongoDB aggregation for cosine similarity
                                   instead of Atlas vector search
            
        Returns:
            List of (course_dict, similarity_score) tuples
        """
        if not self.connected:
            raise ConnectionError("Not connected to MongoDB")
        
        # Use cosine similarity aggregation if requested
        if use_cosine_aggregation:
            logger.info("Using MongoDB aggregation for cosine similarity search")
            return self.cosine_similarity_search(query_vector, top_k, vector_field)
        
        query_vector_list = query_vector.tolist()
        
        try:
            # Try Atlas vector search first
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": vector_field,
                        "queryVector": query_vector_list,
                        "numCandidates": limit,
                        "limit": top_k
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        "code": 1,
                        "title": 1,
                        "description": 1,
                        "prerequisites": 1,
                        "corequisites": 1,
                        "credits": 1,
                        "distribution": 1,
                        "url": 1,
                        "full_text": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]
            
            results = list(self.collection.aggregate(pipeline))
            
            # Format results
            formatted_results = []
            for doc in results:
                course = {k: v for k, v in doc.items() 
                         if k not in ['_id', 'score', 'embedding']}
                score = doc.get('score', 0.0)
                formatted_results.append((course, float(score)))
            
            return formatted_results
            
        except OperationFailure:
            # Fallback to cosine similarity aggregation
            logger.info("Atlas vector search not available, using cosine similarity aggregation")
            return self.cosine_similarity_search(query_vector, top_k, vector_field)
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            # Fallback to cosine similarity aggregation
            logger.info("Falling back to cosine similarity aggregation")
            return self.cosine_similarity_search(query_vector, top_k, vector_field)
    
    def cosine_similarity_search(self,
                                query_vector: np.ndarray,
                                top_k: int = 5,
                                vector_field: str = 'embedding') -> List[Tuple[Dict, float]]:
        """
        Perform cosine similarity search using MongoDB aggregation pipeline.
        Computes cosine similarity directly in MongoDB for better performance.
        
        Args:
            query_vector: Query vector as numpy array
            top_k: Number of top results to return
            vector_field: Name of the field containing the embedding
            
        Returns:
            List of (course_dict, similarity_score) tuples
        """
        if not self.connected:
            raise ConnectionError("Not connected to MongoDB")
        
        query_vector_list = query_vector.tolist()
        query_norm = np.linalg.norm(query_vector)
        
        if query_norm == 0:
            return []
        
        try:
            # MongoDB aggregation pipeline to compute cosine similarity
            pipeline = [
                # Match documents that have embeddings
                {"$match": {vector_field: {"$exists": True}}},
                
                # Add computed fields for cosine similarity
                {
                    "$addFields": {
                        # Compute dot product: sum(query[i] * embedding[i])
                        "dot_product": {
                            "$reduce": {
                                "input": {"$range": [0, {"$size": f"${vector_field}"}]},
                                "initialValue": 0,
                                "in": {
                                    "$add": [
                                        "$$value",
                                        {
                                            "$multiply": [
                                                {"$arrayElemAt": [query_vector_list, "$$this"]},
                                                {"$arrayElemAt": [f"${vector_field}", "$$this"]}
                                            ]
                                        }
                                    ]
                                }
                            }
                        },
                        # Compute embedding norm: sqrt(sum(embedding[i]^2))
                        "embedding_norm": {
                            "$sqrt": {
                                "$reduce": {
                                    "input": f"${vector_field}",
                                    "initialValue": 0,
                                    "in": {
                                        "$add": [
                                            "$$value",
                                            {"$multiply": ["$$this", "$$this"]}
                                        ]
                                    }
                                }
                            }
                        }
                    }
                },
                
                # Compute cosine similarity
                {
                    "$addFields": {
                        "similarity": {
                            "$cond": {
                                "if": {"$eq": ["$embedding_norm", 0]},
                                "then": 0,
                                "else": {
                                    "$divide": [
                                        "$dot_product",
                                        {"$multiply": [query_norm, "$embedding_norm"]}
                                    ]
                                }
                            }
                        }
                    }
                },
                
                # Project only needed fields
                {
                    "$project": {
                        "_id": 0,
                        "code": 1,
                        "title": 1,
                        "description": 1,
                        "prerequisites": 1,
                        "corequisites": 1,
                        "credits": 1,
                        "distribution": 1,
                        "url": 1,
                        "similarity": 1
                    }
                },
                
                # Sort by similarity descending
                {"$sort": {"similarity": -1}},
                
                # Limit to top k
                {"$limit": top_k}
            ]
            
            results = list(self.collection.aggregate(pipeline))
            
            # Format results
            formatted_results = []
            for doc in results:
                course = {k: v for k, v in doc.items() if k != 'similarity'}
                similarity = doc.get('similarity', 0.0)
                formatted_results.append((course, float(similarity)))
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in cosine similarity search: {e}")
            # Fallback to manual search
            logger.info("Falling back to manual vector search")
            return self._manual_vector_search(query_vector, top_k, vector_field)
    
    def _manual_vector_search(self,
                              query_vector: np.ndarray,
                              top_k: int = 5,
                              vector_field: str = 'embedding') -> List[Tuple[Dict, float]]:
        """
        Manual vector search using cosine similarity (Python-based).
        Used as fallback when MongoDB aggregation fails.
        
        Args:
            query_vector: Query vector as numpy array
            top_k: Number of top results to return
            vector_field: Name of the field containing the embedding
            
        Returns:
            List of (course_dict, similarity_score) tuples
        """
        # Get all documents with embeddings
        cursor = self.collection.find({vector_field: {"$exists": True}})
        
        similarities = []
        query_norm = np.linalg.norm(query_vector)
        
        if query_norm == 0:
            return []
        
        for doc in cursor:
            embedding = np.array(doc[vector_field])
            
            # Calculate cosine similarity
            dot_product = np.dot(query_vector, embedding)
            embedding_norm = np.linalg.norm(embedding)
            
            if embedding_norm == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (query_norm * embedding_norm)
            
            # Create course dict without embedding
            course = {k: v for k, v in doc.items() 
                     if k not in ['_id', vector_field]}
            # Note: full_text is not stored in MongoDB
            similarities.append((course, float(similarity)))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_course(self, course_code: str) -> Optional[Dict]:
        """
        Get a course by code.
        
        Args:
            course_code: Course code (e.g., "CPTR 141")
            
        Returns:
            Course dictionary or None if not found
        """
        if not self.connected:
            raise ConnectionError("Not connected to MongoDB")
        
        course = self.collection.find_one({'code': course_code})
        if course:
            # Remove embedding and _id for cleaner output
            course.pop('_id', None)
            course.pop('embedding', None)
            # Note: full_text is not stored in MongoDB
        return course
    
    def get_all_courses(self) -> List[Dict]:
        """
        Get all courses from MongoDB.
        
        Returns:
            List of course dictionaries
        """
        if not self.connected:
            raise ConnectionError("Not connected to MongoDB")
        
        courses = []
        for doc in self.collection.find({}):
            # Remove embedding and _id
            doc.pop('_id', None)
            doc.pop('embedding', None)
            # Note: full_text is not stored in MongoDB
            courses.append(doc)
        
        return courses
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        if not self.connected:
            raise ConnectionError("Not connected to MongoDB")
        
        result = self.collection.delete_many({})
        logger.info(f"Deleted {result.deleted_count} documents")
        return result.deleted_count
    
    def count_courses(self) -> int:
        """Get the number of courses in the collection."""
        if not self.connected:
            raise ConnectionError("Not connected to MongoDB")
        
        return self.collection.count_documents({})
    
    def store_word_vectors(self, model) -> bool:
        """
        Store Word2Vec vocabulary and word vectors in MongoDB.
        
        Args:
            model: Trained Word2Vec model
            
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            raise ConnectionError("Not connected to MongoDB")
        
        try:
            logger.info("Storing Word2Vec vocabulary in MongoDB...")
            
            # Clear existing vocabulary
            self.vocab_collection.delete_many({})
            
            # Store vocabulary metadata
            vocab_metadata = {
                '_id': 'vocab_metadata',
                'vector_size': model.wv.vector_size,
                'vocab_size': len(model.wv.key_to_index),
                'index_to_key': list(model.wv.index_to_key)
            }
            self.vocab_collection.replace_one(
                {'_id': 'vocab_metadata'},
                vocab_metadata,
                upsert=True
            )
            
            # Store word vectors in batches
            batch_size = 1000
            words = list(model.wv.key_to_index.keys())
            total_batches = (len(words) + batch_size - 1) // batch_size
            
            for i in range(0, len(words), batch_size):
                batch_words = words[i:i + batch_size]
                batch_docs = []
                
                for word in batch_words:
                    vector = model.wv[word].tolist()
                    batch_docs.append({
                        '_id': word,
                        'word': word,
                        'vector': vector
                    })
                
                if batch_docs:
                    # Use bulk write for efficiency
                    operations = [
                        ReplaceOne({'_id': doc['_id']}, doc, upsert=True)
                        for doc in batch_docs
                    ]
                    self.vocab_collection.bulk_write(operations, ordered=False)
                
                batch_num = (i // batch_size) + 1
                logger.info(
                    f"Stored batch {batch_num}/{total_batches} "
                    f"({len(batch_words)} words)"
                )
            
            logger.info(
                f"Successfully stored {len(words)} word vectors in MongoDB"
            )
            return True
            
        except Exception as e:
            logger.error(f"Error storing word vectors: {e}")
            return False
    
    def has_word_vectors(self) -> bool:
        """Check if word vectors are stored in MongoDB."""
        if not self.connected:
            return False
        
        try:
            metadata = self.vocab_collection.find_one({'_id': 'vocab_metadata'})
            return metadata is not None
        except Exception:
            return False
    
    def get_vector_size(self) -> Optional[int]:
        """Get the vector size from stored vocabulary metadata."""
        if not self.connected:
            return None
        
        try:
            metadata = self.vocab_collection.find_one({'_id': 'vocab_metadata'})
            if metadata:
                return metadata.get('vector_size')
        except Exception:
            pass
        return None
    
    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """
        Get word vector from MongoDB.
        
        Args:
            word: Word to look up
            
        Returns:
            Word vector as numpy array, or None if not found
        """
        if not self.connected:
            return None
        
        try:
            doc = self.vocab_collection.find_one({'_id': word})
            if doc and 'vector' in doc:
                return np.array(doc['vector'])
        except Exception as e:
            logger.warning(f"Error retrieving word vector for '{word}': {e}")
        
        return None
    
    def get_word_vectors_batch(self, words: List[str]) -> Dict[str, np.ndarray]:
        """
        Get multiple word vectors from MongoDB in a single query.
        
        Args:
            words: List of words to look up
            
        Returns:
            Dictionary mapping words to their vectors
        """
        if not self.connected:
            return {}
        
        try:
            docs = self.vocab_collection.find({'_id': {'$in': words}})
            result = {}
            for doc in docs:
                if 'vector' in doc:
                    result[doc['word']] = np.array(doc['vector'])
            return result
        except Exception as e:
            logger.warning(f"Error retrieving word vectors batch: {e}")
            return {}


if __name__ == '__main__':
    # Example usage
    storage = MongoDBStorage()
    if storage.connect():
        print(f"Connected! Collection has {storage.count_courses()} courses")
        storage.disconnect()
