import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from feature_extractor import (
    calculate_circularity, calculate_shape_descriptors, fourier_boundary_descriptor, major_color, remove_background, compute_hog_with_opencv, compute_sift_features, 
    compute_gist_like_features, extract_fourier_feature_vector,
    compute_wavelet_features, compute_dct_features
)


class SIFT_BoW_Converter:
    def __init__(self, n_clusters=100, bow_path="bow_model.pkl"):
        self.n_clusters = n_clusters
        self.bow_path = bow_path
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0)
        self.is_trained = False
        
        # Load pre-trained model if exists
        if os.path.exists(self.bow_path):
            self.load()
        
    def train(self, train_image_dir: str, background=False):
        """
        Train BoW codebook using images from a directory
        """
        # Collect descriptors from training images
        all_descriptors = []
        supported_exts = ('.jpg', '.jpeg', '.png', '.webp')
        
        image_paths = [
            os.path.join(train_image_dir, f) 
            for f in os.listdir(train_image_dir)
            if f.lower().endswith(supported_exts)
        ]
        
        print(f"Training BoW model with {len(image_paths)} images...")
        
        for path in tqdm(image_paths, desc="Processing images"):
            img = cv2.imread(path)
            if not background:
                img = remove_background(img)
            _, descriptors = compute_sift_features(img)
            if descriptors is not None:
                all_descriptors.append(descriptors)
        
        if not all_descriptors:
            raise ValueError("No SIFT descriptors found in training images")
            
        # Train KMeans
        all_descriptors = np.vstack(all_descriptors)
        self.kmeans.fit(all_descriptors)
        self.is_trained = True
        
        # Save trained model
        self.save()
        
    def save(self):
        """Save trained BoW model to disk"""
        with open(self.bow_path, 'wb') as f:
            pickle.dump({
                'kmeans': self.kmeans,
                'is_trained': self.is_trained
            }, f)
            
    def load(self):
        """Load pre-trained BoW model from disk"""
        with open(self.bow_path, 'rb') as f:
            data = pickle.load(f)
            self.kmeans = data['kmeans']
            self.is_trained = data['is_trained']
            
    def sift_to_bow(self, descriptors):
        """Convert SIFT descriptors to BoW histogram"""
        if not self.is_trained:
            raise RuntimeError("BoW model not trained! Call train() first")
            
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.n_clusters)
            
        visual_words = self.kmeans.predict(descriptors)
        histogram = np.bincount(visual_words, minlength=self.n_clusters)
        return normalize(histogram.reshape(1, -1)).flatten()


class SimilaritySystem:
    def extract_features(self, image, remove_background_flag=True):
        """Extract and combine all features for an image"""
        if remove_background_flag:
            image = remove_background(image)
        
        # Existing features
        hog_features = compute_hog_with_opencv(image)
        gist_features = compute_gist_like_features(image)
        
        # Spectral features
        fourier_features = extract_fourier_feature_vector(image)
        wavelet_features = compute_wavelet_features(image)
        dct_features = compute_dct_features(image)
        
        # boundary descriptor
        boundary_features = fourier_boundary_descriptor(image, num_descriptors=128)
        
        # Shape descriptors (uncomment if needed)
        major_color_feature = np.sort(major_color(image, 3), axis=0).flatten() if np.sort(major_color(image, 3), axis=0) is not None else np.zeros(9)
        # circularity_features = calculate_circularity(image)
        # shape_descriptors = calculate_shape_descriptors(image)

        # Normalize and flatten all features
        features = [
            normalize(hog_features.reshape(1, -1)).flatten(),
            normalize(gist_features.reshape(1, -1)).flatten(),
            normalize(fourier_features.reshape(1, -1)).flatten(),
            normalize(wavelet_features.reshape(1, -1)).flatten(),
            normalize(dct_features.reshape(1, -1)).flatten(),
            normalize(boundary_features.reshape(1, -1)).flatten(),
            normalize(major_color_feature.reshape(1, -1)).flatten(),
            # normalize(np.array(circularity_features).reshape(1, -1)).flatten(),
            # normalize(np.array(shape_descriptors).reshape(1, -1)).flatten()
        ]

        # Concatenate all features
        combined = np.concatenate(features)
        return combined

    def cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)