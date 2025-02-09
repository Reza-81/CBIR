import os
import cv2
import sqlite3
import numpy as np
import faiss
import pickle
from tqdm import tqdm
from typing import List, Tuple

class DatabaseHandler:
    def __init__(self, db_path: str = "cbir.db"):
        """
        Initialize the CBIR database handler with FAISS index and SQLite metadata storage.
        
        Args:
            db_path (str): Path to SQLite database file
            feature_dim (int): Dimensionality of feature vectors
        """
        self.db_path = db_path
        
        # Initialize SQLite connection
        self.conn = sqlite3.connect(db_path)
        self._create_tables()
        
        # Load existing index or prepare for new one
        self._initialize_faiss()

    def _create_tables(self):
        """Create database tables if they don't exist"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                label TEXT,
                path TEXT UNIQUE NOT NULL,
                features BLOB NOT NULL
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_path ON images (path)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_label ON images (label)")
        self.conn.commit()
    
    def _initialize_faiss(self):
        """Initialize FAISS index from existing data or create empty"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT features FROM images LIMIT 1")
        result = cursor.fetchone()
        
        if result:
            # Database has existing features - load full index
            self.feature_dim = len(pickle.loads(result[0]))
            self.index = faiss.IndexFlatIP(self.feature_dim)
            self._load_faiss_index()
        else:
            # New database - index will be created on first add
            self.index = None
            self.feature_dim = None

    def _load_faiss_index(self):
        """Load existing features into FAISS index"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT features FROM images")
        features = [pickle.loads(row[0]) for row in cursor.fetchall()]
        
        if features:
            self.index.add(np.array(features).astype('float32'))

    def add_image(self, image_path: str, feature_extractor, label: str = None) -> bool:
        """
        Add a single image to the database if it doesn't exist
        
        Args:
            image_path (str): Path to image file
            label (str): Optional label for the image
            feature_extractor: Your feature extraction function
            
        Returns:
            bool: True if image was added, False if already exists
        """
        if self.image_exists(image_path):
            return False

        # Extract features
        img = cv2.imread(image_path)
        features = feature_extractor.extract_features(img)

        if self.index is None:
            # Initialize FAISS index with first feature's dimension
            self.feature_dim = features.shape[0]
            self.index = faiss.IndexFlatIP(self.feature_dim)
            print(f"Initialized FAISS index with dimension {self.feature_dim}")
        
        # Verify feature dimensions
        if features.shape[0] != self.feature_dim:
            raise ValueError(f"Feature dimension mismatch. Expected {self.feature_dim}, got {features.shape[0]}")
        
        # Store in database
        serialized = pickle.dumps(features)
        self.conn.execute(
            "INSERT INTO images (name, label, path, features) VALUES (?, ?, ?, ?)",
            (os.path.basename(image_path), label, image_path, serialized)
        )
        self.conn.commit()
        
        # Add to FAISS index
        self.index.add(np.array([features]).astype('float32'))
        return True

    def add_directory(self, dir_path: str, feature_extractor, label: str = None):
        """
        Add all images in a directory to the database
        
        Args:
            dir_path (str): Path to directory containing images
            label (str): Optional label for all images in directory
            feature_extractor: Your feature extraction function
        """
        supported_exts = ('.jpg', '.jpeg', '.png', '.webp')

        image_paths = [
            os.path.join(dir_path, f) for f in os.listdir(dir_path)
            if f.lower().endswith(supported_exts)
        ]

        for path in tqdm(image_paths, desc="Processing images"):
            self.add_image(path, feature_extractor, label)

    def search(self, query_image: np.ndarray, feature_extractor, k: int = 10) -> List[Tuple]:
        """
        Search for similar images in the database
        
        Args:
            query_image (np.ndarray): Query image as numpy array
            feature_extractor: Your feature extraction function (SimilaritySystem instance)
            k (int): Number of results to return
            
        Returns:
            List[Tuple]: List of (image_path, similarity_score) tuples
        """
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("Database is empty! Add images before searching.")
        
        # Extract query features
        query_features = feature_extractor.extract_features(query_image)
        
        # FAISS search
        distances, indices = self.index.search(
            np.array([query_features]).astype('float32'), k
        )
        
        # Get metadata from SQLite and calculate cosine similarity
        cursor = self.conn.cursor()
        results = []
        for idx, faiss_score in zip(indices[0], distances[0]):
            # Get database record
            cursor.execute("SELECT path, features FROM images WHERE rowid=?", (int(idx)+1,))
            path, features_blob = cursor.fetchone()
            
            # Load stored features
            stored_features = pickle.loads(features_blob)
            
            # Calculate cosine similarity
            cosine_sim = feature_extractor.cosine_similarity(query_features, stored_features)
            
            # Print comparison (FAISS score vs manual calculation)
            print(f"Image: {os.path.basename(path)}")
            print(f"  FAISS Score: {faiss_score:.4f} | Cosine Similarity: {cosine_sim:.4f}")
            print("-" * 50)
            
            # Maintain original return format
            results.append((path, float(faiss_score)))
        print('=' * 50)
            
        return results

    def delete_image(self, image_path: str):
        """
        Delete an image from the database
        
        Args:
            image_path (str): Path of image to delete
        """
        # Get the database ID
        cursor = self.conn.cursor()
        cursor.execute("SELECT rowid FROM images WHERE path=?", (image_path,))
        result = cursor.fetchone()
        
        if not result:
            raise ValueError("Image not found in database")
            
        db_id = result[0]
        
        # Delete from SQLite
        self.conn.execute("DELETE FROM images WHERE path=?", (image_path,))
        self.conn.commit()
        
        # Rebuild FAISS index (FAISS doesn't support direct deletion)
        self._rebuild_faiss_index()

    def image_exists(self, image_path: str) -> bool:
        """Check if an image exists in the database"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM images WHERE path=?", (image_path,))
        return cursor.fetchone() is not None

    def _rebuild_faiss_index(self):
        """Rebuild the FAISS index from database"""
        self.index.reset()
        cursor = self.conn.cursor()
        cursor.execute("SELECT features FROM images")
        features = [pickle.loads(row[0]) for row in cursor.fetchall()]
        
        if features:
            self.index.add(np.array(features).astype('float32'))

    def backup(self, backup_path: str):
        """Create a backup of the database"""
        with sqlite3.connect(backup_path) as backup_conn:
            self.conn.backup(backup_conn)
        faiss.write_index(self.index, backup_path + ".index")

    def restore(self, backup_path: str):
        """Restore database from backup"""
        self.conn.close()
        os.replace(backup_path, self.db_path)
        self.index = faiss.read_index(backup_path + ".index")
        self.conn = sqlite3.connect(self.db_path)

    def get_stats(self) -> dict:
        """Get database statistics"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM images")
        count = cursor.fetchone()[0]
        
        return {
            "total_images": count,
            "index_size": self.index.ntotal,
            "feature_dim": self.feature_dim,
            "labels": self.get_all_labels()
        }

    def get_all_labels(self) -> List[str]:
        """Get list of all unique labels"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT label FROM images")
        return [row[0] for row in cursor.fetchall() if row[0] is not None]

    def __del__(self):
        """Cleanup when instance is destroyed"""
        self.conn.close()