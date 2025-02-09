# Content-Based Image Retrieval (CBIR) System

This project implements a Content-Based Image Retrieval system using a graphical interface and a combination of feature extraction, clustering, and database techniques.

![Screenshot of the CBIR System](screenshot.png)


## Overview

## How It Works

- **Feature Extraction:**  
    The system employs multiple feature extraction techniques including:
    - **HOG (Histogram of Oriented Gradients):** Captures texture and shape information.
    - **GIST-like Features:** Summarizes the overall scene structure.
    - **Fourier Descriptors:** Analyzes frequency components.
    - **Wavelet Features:** Provides multi-resolution information.
    - **DCT Features:** Represents image energy distribution.
    - **Shape and Color Descriptors:** Computes circularity, major color, and other boundary descriptors.
    - **SIFT Features and Bag-of-Words (BoW):** Uses SIFT to extract keypoints and convert them into a BoW histogram for efficient matching.

- **Similarity System:**  
    The core of the project interfaces with the feature extraction module. It extracts and normalizes various features for a given image and computes similarity (e.g., using cosine similarity) between images.

- **Database Implementation:**  
    - **SQLite:** Used to store metadata (such as image name, path, and label) and serialized feature vectors.
    - **FAISS Index:** A fast nearest neighbor search library used to efficiently retrieve relevant images based on feature similarity.
    - **Feedback Mechanism:** The system can refine search results based on user feedback by updating the query with features from both relevant and non-relevant images.

- **Relevance Feedback Mechanism:**  
    The relevance feedback mechanism improves search results by incorporating user feedback. The algorithm used is the Rocchio algorithm, which works as follows:
    1. **Initial Search:** The user performs an initial search with a query image.
    2. **Feedback Collection:** The user marks images as relevant or non-relevant from the search results.
    3. **Feature Adjustment:** The system adjusts the query feature vector by emphasizing features from relevant images and de-emphasizing features from non-relevant images.
    4. **Refined Search:** The adjusted query vector is used to perform a new search, yielding more accurate results.
    This iterative process continues until the user is satisfied with the search results.


## Database Details

The SQLite database is handled by the `DatabaseHandler` class. Key points include:
- **Tables & Indexes:**  
    A table named `images` stores each image's ID, name, label, file path, and feature vector. Indexes on the path and label improve query performance.
- **FAISS Integration:**  
    A FAISS index is built or updated every time an image is added/deleted, allowing efficient similarity searches across high-dimensional feature vectors.

## How to Use the Project

1. **Installation:**
     - Ensure Python is installed along with required libraries: Tkinter, OpenCV, PIL, sklearn, FAISS, rembg, and tqdm.
     - Clone the project repository and install dependencies using pip:
         ```
         pip install -r requirements.txt
         ```
2. **Starting the Application:**
     - Run the main application:
         ```
         python main.py
         ```
     - The Tkinter GUI will launch displaying options to add images, search, or refine search results.
3. **Adding Images:**
     - Use the "Add Image" or "Add Directory" buttons in the GUI to add images to the database.
     - Each image's features are extracted and stored in the database.
4. **Searching Images:**
     - Load a query image.
     - Click on "Search" to retrieve the top 10 similar images based on feature similarity.
5. **Refining Search:**
     - Provide feedback using the "Refine Search" button. The system will update the query features based on user-selected relevant/non-relevant images and return updated results.

## Project Structure

- **main.py:**  
    Sets up the Tkinter GUI, initializes backend components, and manages user interactions.
    
- **similarity_system.py:**  
    Contains the `SimilaritySystem` class for image similarity computation and feature extraction orchestration.
    
- **feature_extractor.py:**  
    Implements various feature extraction functions including HOG, GIST, Fourier descriptors, wavelet, and DCT features.
    
- **database.py:**  
    Manages the database connection, persistence of image metadata and features, and fast similarity search using FAISS.

## Contributing

Contributions are welcome. Fork the repository and submit pull requests for improvements, bug fixes, or additional features.
