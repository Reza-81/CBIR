import os
import cv2
import sqlite3
import numpy as np
import faiss
import pickle
from tqdm import tqdm
from typing import List, Tuple

class DatabaseHandler:
    def __init__(self, connection):
        self.conn = connection
        self._create_tables()
        self._initialize_faiss()

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS images (
                name TEXT NOT NULL,
                label TEXT,
                path TEXT UNIQUE NOT NULL,
                features BLOB NOT NULL
            )
        """)
        self.conn.commit()

    def _initialize_faiss(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT features FROM images LIMIT 1")
        result = cursor.fetchone()
        if result:
            self.feature_dim = len(pickle.loads(result[0]))
            self.index = faiss.IndexFlatIP(self.feature_dim)
            self._load_faiss_index()
        else:
            self.index = None
            self.feature_dim = None

    def _load_faiss_index(self):
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
            feature_extractor: Feature extraction function
            label (str): Optional label for the image
            
        Returns:
            bool: True if image was added, False if already exists
        """
        if self.image_exists(image_path):
            return False

        img = cv2.imread(image_path)
        features = feature_extractor.extract_features(img)

        if self.index is None:
            self.feature_dim = features.shape[0]
            self.index = faiss.IndexFlatIP(self.feature_dim)
            print(f"Initialized FAISS index with dimension {self.feature_dim}")
        
        if features.shape[0] != self.feature_dim:
            raise ValueError(f"Feature dimension mismatch. Expected {self.feature_dim}, got {features.shape[0]}")
        
        serialized = pickle.dumps(features)
        self.conn.execute(
            "INSERT INTO images (name, label, path, features) VALUES (?, ?, ?, ?)",
            (os.path.basename(image_path), label, image_path, serialized)
        )
        self.conn.commit()
        
        self.index.add(np.array([features]).astype('float32'))
        return True

    def add_directory(self, dir_path: str, feature_extractor, label: str = None):
        """
        Add all images in a directory to the database
        
        Args:
            dir_path (str): Path to directory containing images
            feature_extractor: Feature extraction function
            label (str): Optional label for all images
        """
        supported_exts = ('.jpg', '.jpeg', '.png', '.webp')
        image_paths = [
            os.path.join(dir_path, f) for f in os.listdir(dir_path)
            if f.lower().endswith(supported_exts)
        ]

        for path in tqdm(image_paths, desc="Processing images"):
            self.add_image(path, feature_extractor, label)

    def delete_image(self, image_path: str):
        """
        Delete an image from the database
        
        Args:
            image_path (str): Path of image to delete
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT rowid FROM images WHERE path=?", (image_path,))
        result = cursor.fetchone()
        
        if not result:
            raise ValueError("Image not found in database")
            
        self.conn.execute("DELETE FROM images WHERE path=?", (image_path,))
        self.conn.commit()
        self._rebuild_faiss_index()

    def image_exists(self, image_path: str) -> bool:
        """Check if an image exists in the database"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT 1 FROM images WHERE path=?", (image_path,))
        return cursor.fetchone() is not None

    def _rebuild_faiss_index(self):
        """Rebuild FAISS index from database"""
        self.index.reset()
        cursor = self.conn.cursor()
        cursor.execute("SELECT features FROM images")
        features = [pickle.loads(row[0]) for row in cursor.fetchall()]
        if features:
            self.index.add(np.array(features).astype('float32'))

    def backup(self, backup_path: str):
        """Create a backup of the database and FAISS index"""
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA database_list")
        db_list = cursor.fetchall()
        main_db = next(row for row in db_list if row[1] == 'main')
        current_db_path = main_db[2]

        with sqlite3.connect(backup_path) as backup_conn:
            self.conn.backup(backup_conn)
        faiss.write_index(self.index, backup_path + ".index")

    def restore(self, backup_path: str):
        """Restore database and FAISS index from backup"""
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA database_list")
        db_list = cursor.fetchall()
        main_db = next(row for row in db_list if row[1] == 'main')
        current_db_path = main_db[2]

        self.conn.close()
        os.replace(backup_path, current_db_path)
        self.conn = sqlite3.connect(current_db_path)
        self.index = faiss.read_index(backup_path + ".index")
        self._initialize_faiss()

    def get_stats(self) -> dict:
        """Get database statistics"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM images")
        count = cursor.fetchone()[0]
        
        return {
            "total_images": count,
            "index_size": self.index.ntotal if self.index else 0,
            "feature_dim": self.feature_dim,
            "labels": self.get_all_labels()
        }

    def get_all_labels(self) -> List[str]:
        """Get list of all unique labels"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT label FROM images")
        return [row[0] for row in cursor.fetchall() if row[0] is not None]

    # Existing new methods below
    def search(self, query_image: np.ndarray, feature_extractor, k: int = 10) -> List[Tuple]:
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("Database is empty! Add images before searching.")
        
        query_features = feature_extractor.extract_features(query_image)
        distances, indices = self.index.search(np.array([query_features]).astype('float32'), k)
        
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < 0:
                continue
            cursor = self.conn.cursor()
            cursor.execute("SELECT path FROM images WHERE rowid=?", (int(idx) + 1,))
            path = cursor.fetchone()[0]
            results.append((path, float(score)))
        return results

    def get_image_path(self, rowid: int) -> str:
        cursor = self.conn.cursor()
        cursor.execute("SELECT path FROM images WHERE rowid=?", (rowid,))
        result = cursor.fetchone()
        return result[0] if result else None