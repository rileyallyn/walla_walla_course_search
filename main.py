"""
Main script to orchestrate scraping, training, and querying of course data.
"""

import os
import sys
import argparse
from dotenv import load_dotenv
from walla_walla_course_search.scraper import CourseScraper
from walla_walla_course_search.word2vec_trainer import Word2VecTrainer
from walla_walla_course_search.semantic_search import SemanticSearch
from walla_walla_course_search.populate_mongodb import populate_mongodb
import logging

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def scrape_courses(base_url: str, output_file: str = 'courses.json'):
    """Scrape courses from the catalog website."""
    logger.info("Starting course scraping...")
    scraper = CourseScraper(base_url, delay=1.0)
    scraper.scrape_recursive(max_depth=3)
    scraper.save_courses(output_file)
    return scraper.get_courses()


def train_model(courses_file: str = 'courses.json', 
                model_file: str = 'word2vec_model.model',
                epochs: int = 100,
                vector_size: int = 100):
    """Train word2vec model on course data."""
    logger.info("Starting model training...")
    
    import json
    with open(courses_file, 'r', encoding='utf-8') as f:
        courses = json.load(f)
    
    trainer = Word2VecTrainer(courses)
    trainer.prepare_training_data()
    trainer.train(epochs=epochs, vector_size=vector_size)
    trainer.save_model(model_file)
    return trainer.get_model()


def interactive_search(model_file: str = 'word2vec_model.model',
                       courses_file: str = 'courses.json',
                       use_mongodb: bool = False,
                       mongodb_uri: str = None,
                       use_cosine_aggregation: bool = False):
    """Interactive semantic search interface."""
    if use_mongodb:
        logger.info("Connecting to MongoDB...")
        # Model is optional if word vectors are stored in MongoDB
        # We'll check this after connecting
    else:
        logger.info("Loading model and courses...")
    
    try:
        search = SemanticSearch(
            model_file, 
            courses_file,
            use_mongodb=use_mongodb,
            mongodb_uri=mongodb_uri,
            use_cosine_aggregation=use_cosine_aggregation
        )
    except (ValueError, FileNotFoundError) as e:
        if use_mongodb:
            logger.error(
                f"Failed to initialize search: {e}\n"
                "MongoDB connection failed and no model file available. "
                "Please either:\n"
                "  1. Fix MongoDB connection, or\n"
                "  2. Train a model first with --train"
            )
        else:
            logger.error(f"Failed to initialize search: {e}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("Walla Walla University Computer Science Course Semantic Search")
    print("="*70)
    if search.use_mongodb:
        search_method = "MongoDB Cosine Similarity Aggregation" if search.use_cosine_aggregation else "MongoDB Atlas Vector Search"
        print(f"Using: {search_method}")
    else:
        print("Using: Local Word2Vec Model")
    print("\nYou can:")
    print("  1. Search for courses semantically")
    print("  2. Ask questions about courses")
    print("  3. Toggle MongoDB (type 'toggle mongodb' or 'toggle local')")
    if search.use_mongodb:
        print("  4. Toggle search method (type 'toggle cosine' or 'toggle atlas')")
    print("\nSearch examples:")
    print("  - 'machine learning'")
    print("  - 'web development'")
    print("  - 'databases'")
    print("  - 'similar to CPTR 141'")
    print("\nQuestion examples:")
    print("  - 'What classes do I need before taking Compilers and Languages?'")
    print("  - 'What is CPTR 354 about?'")
    print("  - 'How many credits is Software Engineering?'")
    print("  - 'What classes do I take with CPTR 450?'")
    print("\nType 'quit' or 'exit' to exit.\n")
    
    while True:
        try:
            query = input("Query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            # Handle toggle commands
            if query.lower().startswith('toggle mongodb'):
                if search.toggle_mongodb(True, mongodb_uri):
                    method = "cosine similarity aggregation" if search.use_cosine_aggregation else "Atlas vector search"
                    print(f"Switched to MongoDB ({method})")
                else:
                    print("Failed to switch to MongoDB")
                continue
            
            if query.lower().startswith('toggle local'):
                if search.toggle_mongodb(False):
                    print("Switched to local model")
                else:
                    print("Failed to switch to local model")
                continue
            
            # Handle cosine aggregation toggle
            if query.lower().startswith('toggle cosine'):
                if search.toggle_cosine_aggregation(True):
                    print("Switched to MongoDB cosine similarity aggregation")
                else:
                    print("Failed to switch to cosine aggregation")
                continue
            
            if query.lower().startswith('toggle atlas'):
                if search.toggle_cosine_aggregation(False):
                    print("Switched to MongoDB Atlas vector search")
                else:
                    print("Failed to switch to Atlas search")
                continue
            
            # Check if query is a question (contains question words or patterns)
            is_question = any(
                query.lower().startswith(qword) or qword in query.lower()
                for qword in ['what', 'how', 'when', 'where', 'who', 'which',
                             'tell me', 'describe', 'explain', 'need before',
                             'prerequisite', 'corequisite']
            )
            
            if is_question:
                # Use question answering
                answer = search.answer_question(query)
                print(f"\n{answer}\n")
                print("-" * 70 + "\n")
            elif query.lower().startswith('similar to'):
                # Find similar courses
                course_code = query.replace('similar to', '').strip().upper()
                results = search.find_similar_courses(course_code, top_k=5)
                print(f"\nCourses similar to {course_code}:")
                if not results:
                    print("No results found.")
                else:
                    print("-" * 70)
                    for i, (course, score) in enumerate(results, 1):
                        code = course.get('code', 'N/A')
                        title = course.get('title', 'N/A')
                        print(f"\n{i}. {code} - {title}")
                        print(f"   Similarity: {score:.4f}")
                        
                        if course.get('description'):
                            desc = course['description']
                            if len(desc) > 200:
                                desc = desc[:200] + '...'
                            print(f"   Description: {desc}")
                print("\n" + "-" * 70 + "\n")
            else:
                # Regular semantic search
                results = search.search(query, top_k=5)
                print(f"\nResults for '{query}':")
                if not results:
                    print("No results found.")
                else:
                    print("-" * 70)
                    for i, (course, score) in enumerate(results, 1):
                        code = course.get('code', 'N/A')
                        title = course.get('title', 'N/A')
                        print(f"\n{i}. {code} - {title}")
                        print(f"   Similarity: {score:.4f}")
                        
                        if course.get('description'):
                            desc = course['description']
                            if len(desc) > 200:
                                desc = desc[:200] + '...'
                            print(f"   Description: {desc}")
                        
                        if course.get('url'):
                            print(f"   URL: {course['url']}")
                print("\n" + "-" * 70 + "\n")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error during search: {e}")
            print(f"Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Semantic course search using word2vec'
    )
    parser.add_argument(
        '--scrape',
        action='store_true',
        help='Scrape courses from the website'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train word2vec model'
    )
    parser.add_argument(
        '--search',
        action='store_true',
        help='Start interactive search interface'
    )
    parser.add_argument(
        '--query',
        type=str,
        help='Run a single query (non-interactive)'
    )
    parser.add_argument(
        '--ask',
        type=str,
        help='Ask a question about a course (e.g., "What classes do I need before taking Compilers and Languages?")'
    )
    parser.add_argument(
        '--base-url',
        type=str,
        default='https://wallawalla.smartcatalogiq.com/current/undergraduate-bulletin/courses/cptr-computer-science',
        help='Base URL to scrape'
    )
    parser.add_argument(
        '--courses-file',
        type=str,
        default='courses.json',
        help='Path to courses JSON file'
    )
    parser.add_argument(
        '--model-file',
        type=str,
        default='word2vec_model.model',
        help='Path to word2vec model file'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--vector-size',
        type=int,
        default=100,
        help='Word vector dimensionality'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run scrape, train, and search in sequence'
    )
    parser.add_argument(
        '--populate-mongodb',
        action='store_true',
        help='Populate MongoDB with course embeddings'
    )
    parser.add_argument(
        '--mongodb-uri',
        type=str,
        default=None,
        help='MongoDB connection string (or set MONGODB_URI in .env file)'
    )
    parser.add_argument(
        '--use-mongodb',
        action='store_true',
        help='Use MongoDB for vector search instead of local model'
    )
    parser.add_argument(
        '--use-cosine-aggregation',
        action='store_true',
        help='Use MongoDB aggregation for cosine similarity instead of Atlas vector search'
    )
    parser.add_argument(
        '--mongodb-db',
        type=str,
        default=None,
        help='MongoDB database name (or set MONGODB_DATABASE in .env file)'
    )
    parser.add_argument(
        '--mongodb-collection',
        type=str,
        default=None,
        help='MongoDB collection name (or set MONGODB_COLLECTION in .env file)'
    )
    
    args = parser.parse_args()
    
    # If --all is specified, run everything
    if args.all:
        args.scrape = True
        args.train = True
        args.search = True
    
    # Populate MongoDB
    if args.populate_mongodb:
        if not os.path.exists(args.model_file):
            logger.error(f"Model file not found: {args.model_file}")
            logger.error("Please run with --train first")
            sys.exit(1)
        if not os.path.exists(args.courses_file):
            logger.error(f"Courses file not found: {args.courses_file}")
            logger.error("Please run with --scrape first")
            sys.exit(1)
        populate_mongodb(
            model_path=args.model_file,
            courses_path=args.courses_file,
            mongodb_uri=args.mongodb_uri or os.getenv('MONGODB_URI'),
            database_name=args.mongodb_db or os.getenv('MONGODB_DATABASE', 'course_search'),
            collection_name=args.mongodb_collection or os.getenv('MONGODB_COLLECTION', 'courses'),
            clear_existing=False
        )
    
    # Run scraping
    if args.scrape:
        scrape_courses(args.base_url, args.courses_file)
    
    # Train model
    if args.train:
        if not os.path.exists(args.courses_file):
            logger.error(f"Courses file not found: {args.courses_file}")
            logger.error("Please run with --scrape first")
            sys.exit(1)
        train_model(args.courses_file, args.model_file, args.epochs, args.vector_size)
    
    # Run search
    if args.query:
        # Model only required if not using MongoDB
        if not args.use_mongodb and not os.path.exists(args.model_file):
            logger.error(f"Model file not found: {args.model_file}")
            logger.error("Please run with --train first, or use --use-mongodb")
            sys.exit(1)
        search = SemanticSearch(
            args.model_file if os.path.exists(args.model_file) else None, 
            args.courses_file,
            use_mongodb=args.use_mongodb,
            mongodb_uri=args.mongodb_uri or os.getenv('MONGODB_URI'),
            mongodb_db=args.mongodb_db or os.getenv('MONGODB_DATABASE', 'course_search'),
            mongodb_collection=args.mongodb_collection or os.getenv('MONGODB_COLLECTION', 'courses'),
            use_cosine_aggregation=args.use_cosine_aggregation
        )
        results = search.search(args.query, top_k=5)
        print(f"\nResults for '{args.query}':\n")
        for i, (course, score) in enumerate(results, 1):
            print(f"{i}. {course.get('code', 'N/A')} - {course.get('title', 'N/A')}")
            print(f"   Similarity: {score:.4f}")
            if course.get('description'):
                desc = course['description'][:150] + '...' if len(course['description']) > 150 else course['description']
                print(f"   {desc}\n")
    
    # Answer a question
    if args.ask:
        # Model only required if not using MongoDB
        if not args.use_mongodb and not os.path.exists(args.model_file):
            logger.error(f"Model file not found: {args.model_file}")
            logger.error("Please run with --train first, or use --use-mongodb")
            sys.exit(1)
        search = SemanticSearch(
            args.model_file if os.path.exists(args.model_file) else None, 
            args.courses_file,
            use_mongodb=args.use_mongodb,
            mongodb_uri=args.mongodb_uri or os.getenv('MONGODB_URI'),
            mongodb_db=args.mongodb_db or os.getenv('MONGODB_DATABASE', 'course_search'),
            mongodb_collection=args.mongodb_collection or os.getenv('MONGODB_COLLECTION', 'courses'),
            use_cosine_aggregation=args.use_cosine_aggregation
        )
        answer = search.answer_question(args.ask)
        print(f"\nQuestion: {args.ask}\n")
        print(f"Answer: {answer}\n")
    
    if args.search:
        # Model only required if not using MongoDB
        if not args.use_mongodb and not os.path.exists(args.model_file):
            logger.error(f"Model file not found: {args.model_file}")
            logger.error("Please run with --train first, or use --use-mongodb")
            sys.exit(1)
        interactive_search(
            args.model_file if os.path.exists(args.model_file) else None, 
            args.courses_file,
            use_mongodb=args.use_mongodb,
            mongodb_uri=args.mongodb_uri or os.getenv('MONGODB_URI'),
            use_cosine_aggregation=args.use_cosine_aggregation
        )
    
    # If no actions specified, show help
    if not any([
        args.scrape, args.train, args.search, args.query, 
        args.ask, args.populate_mongodb, args.all
    ]):
        parser.print_help()


if __name__ == '__main__':
    main()

