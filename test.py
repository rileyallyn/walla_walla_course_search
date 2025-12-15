import json
import sys
import numpy as np
import pymongo
import gensim.downloader as api
import gensim.utils
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# 1. SETUP & LOAD MODEL
# ---------------------------------------------------------
# Load model name from environment variable
MODEL_NAME = os.getenv('WORD2VEC_MODEL', 'word2vec-google-news-300')
print(
    f"Loading Word2Vec model '{MODEL_NAME}'... "
    "(This creates a 1.6GB file in ~/gensim-data)"
)
w2v_model = api.load(MODEL_NAME)


def text_to_vector(text, model):
    """
    Averages word vectors to create a single document vector.
    """
    if not text:
        return np.zeros(model.vector_size).tolist()
        
    words = gensim.utils.simple_preprocess(text)
    word_vectors = [model[word] for word in words if word in model]
    
    if not word_vectors:
        return np.zeros(model.vector_size).tolist()
    
    return np.mean(word_vectors, axis=0).tolist()


# 2. CONNECT TO MONGODB
# ---------------------------------------------------------
# Load MongoDB connection details from environment variables
MONGO_URI = os.getenv('MONGODB_URI') or os.getenv('MONGODB_CONNECTION_STRING')
if not MONGO_URI:
    raise ValueError(
        "MongoDB URI not found. Please set MONGODB_URI in your .env file "
        "or as an environment variable."
    )

DATABASE_NAME = os.getenv('MONGODB_DATABASE', 'course_search')
COLLECTION_NAME = os.getenv('MONGODB_COLLECTION', 'courses')

client = pymongo.MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# Optional: Clear existing data to avoid duplicates during testing
# Set CLEAR_COLLECTION=true in .env to enable
if os.getenv('CLEAR_COLLECTION', 'false').lower() == 'true':
    print("Clearing existing collection...")
    collection.delete_many({})
    print("Collection cleared.")

# 3. CHECK MONGODB & PROCESS FILE & INGEST
# ---------------------------------------------------------
FILENAME = os.getenv('COURSES_FILE', 'courses.json')

# Check if courses already exist in MongoDB
existing_count = collection.count_documents({})
print(f"\nMongoDB Status: {existing_count} courses already in collection")

if os.path.exists(FILENAME):
    print(f"Reading {FILENAME}...")
    
    with open(FILENAME, 'r', encoding='utf-8') as f:
        courses_data = json.load(f)
    
    # Check which courses need to be uploaded
    if existing_count > 0:
        # Get existing course codes (more efficient query)
        existing_codes = set()
        for doc in collection.find({}, {'code': 1, '_id': 0}):
            if doc.get('code'):
                existing_codes.add(doc['code'].upper())
        
        print(f"Found {len(existing_codes)} existing course codes in MongoDB")
        
        # Filter out courses that already exist
        new_courses = [
            course for course in courses_data
            if course.get('code', '').upper() not in existing_codes
        ]
        
        if new_courses:
            skipped = len(courses_data) - len(new_courses)
            print(
                f"Found {len(new_courses)} new courses to upload "
                f"(skipping {skipped} existing)"
            )
            courses_to_process = new_courses
        else:
            print(f"All {len(courses_data)} courses already exist in MongoDB. "
                  "Skipping upload.")
            courses_to_process = []
    else:
        print(f"Processing {len(courses_data)} courses for upload...")
        courses_to_process = courses_data
    
    documents_to_upload = []

    for course in courses_to_process:
        # Create the vector context (Title + Description)
        # Use .get() to avoid errors if fields are missing
        title = course.get('title', '')
        desc = course.get('description', '')
        content_to_embed = f"{title} {desc}"
        
        vector = text_to_vector(content_to_embed, w2v_model)

        # Create clean document
        clean_doc = course.copy()
        
        # Remove 'full_text' if it exists
        clean_doc.pop('full_text', None)
        
        # Add the vector embedding
        clean_doc['embedding'] = vector
        
        documents_to_upload.append(clean_doc)

    # Bulk Upload
    if documents_to_upload:
        result = collection.insert_many(documents_to_upload)
        print(f"Successfully uploaded {len(result.inserted_ids)} courses.")
        final_count = collection.count_documents({})
        print(f"Total courses in MongoDB: {final_count}")
    else:
        print("No new courses to upload.")

else:
    print(f"Error: {FILENAME} not found.")

# 4. SEARCH FUNCTION
# ---------------------------------------------------------
def search_courses(query, model, collection, vector_index=None,
                   num_candidates=None, search_limit=None):
    """
    Search for courses using vector similarity.
    
    Args:
        query: Search query text
        model: Word2Vec model for generating query vectors
        collection: MongoDB collection
        vector_index: Vector search index name (default from env)
        num_candidates: Number of candidates (default from env)
        search_limit: Number of results to return (default from env)
    
    Returns:
        List of course documents with similarity scores
    """
    query_vector = text_to_vector(query, model)
    
    # Get search parameters from arguments or environment variables
    if vector_index is None:
        vector_index = os.getenv('MONGODB_VECTOR_INDEX', 'vector_index')
    if num_candidates is None:
        num_candidates = int(os.getenv('VECTOR_SEARCH_CANDIDATES', '50'))
    if search_limit is None:
        search_limit = int(os.getenv('SEARCH_LIMIT', '5'))
    
    try:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": vector_index,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": num_candidates,
                    "limit": search_limit
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "code": 1,
                    "title": 1,
                    "description": 1,
                    "credits": 1,
                    "prerequisites": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        return list(collection.aggregate(pipeline))
    except Exception as e:
        print(f"Vector search error: {e}")
        print("Falling back to text search...")
        # Fallback to simple text search
        return list(collection.find(
            {
                "$or": [
                    {"title": {"$regex": query, "$options": "i"}},
                    {"description": {"$regex": query, "$options": "i"}},
                    {"code": {"$regex": query, "$options": "i"}}
                ]
            },
            {
                "_id": 0,
                "code": 1,
                "title": 1,
                "description": 1,
                "credits": 1,
                "prerequisites": 1
            }
        ).limit(search_limit))


# 4. INTERACTIVE SEARCH
# ---------------------------------------------------------
print("\n" + "="*70)
print("Interactive Course Search")
print("="*70)
total_courses = collection.count_documents({})
print(f"Total courses in database: {total_courses}")

if total_courses == 0:
    print("\n⚠️  Warning: No courses found in database!")
    print("Please upload courses first by running this script with courses.json")
    print("="*70)
    # Exit early if no courses
    sys.exit(0)

print("\nEnter search queries to find similar courses.")
print("Commands: 'quit' or 'exit' to stop, 'help' for more info")
print("="*70 + "\n")

# Get search parameters from environment variables
vector_index = os.getenv('MONGODB_VECTOR_INDEX', 'vector_index')
num_candidates = int(os.getenv('VECTOR_SEARCH_CANDIDATES', '50'))
search_limit = int(os.getenv('SEARCH_LIMIT', '5'))

while True:
    try:
        # Get query from user
        query = input("Search query: ").strip()
        
        # Check for exit commands
        if query.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        # Check for help command
        if query.lower() in ['help', 'h']:
            print("\n" + "="*70)
            print("Search Help")
            print("="*70)
            print("Enter any text query to search for similar courses.")
            print("Examples:")
            print("  - 'machine learning'")
            print("  - 'web development'")
            print("  - 'databases'")
            print("  - 'programming fundamentals'")
            print("\nCommands:")
            print("  - 'quit' or 'exit' - Stop searching")
            print("  - 'help' - Show this help message")
            print("="*70 + "\n")
            continue
        
        if not query:
            print("Please enter a search query (or 'help' for help).")
            continue
        
        # Perform search
        print(f"\nSearching for: '{query}'...")
        results = search_courses(query, w2v_model, collection)
        
        # Display results
        if results:
            print(f"\nFound {len(results)} results:")
            print("-" * 70)
            for i, res in enumerate(results, 1):
                code = res.get('code', 'N/A')
                title = res.get('title', 'N/A')
                score = res.get('score', 0.0)
                description = res.get('description', '')
                credits = res.get('credits', '')
                prerequisites = res.get('prerequisites', '')
                
                print(f"{i}. {code} - {title}")
                if score > 0:
                    print(f"   Similarity: {score:.4f}")
                if credits:
                    print(f"   Credits: {credits}")
                if prerequisites:
                    print(f"   Prerequisites: {prerequisites}")
                if description:
                    # Truncate long descriptions
                    desc = description[:200] + '...' if len(description) > 200 else description
                    print(f"   Description: {desc}")
                print()
        else:
            print("No results found.")
        
        print("-" * 70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        break
    except Exception as e:
        print(f"\nError during search: {e}\n")