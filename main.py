import sqlite3
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
from similarity_system import SimilaritySystem
from database import DatabaseHandler
import cv2
import os
import numpy as np
import hashlib  # ADD THIS AT THE TOP WITH OTHER IMPORTS

# Database configuration
DATABASE = 'cbir.db'  # ADD THIS AT THE TOP WITH OTHER GLOBALS

class CBIRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Content-Based Image Retrieval System")
        
        # Initialize backend components
        self.feature_extractor = SimilaritySystem()
        self.db = DatabaseHandler(connection=sqlite3.connect(DATABASE, check_same_thread=False))
        
        # NEW: Feedback persistence variables
        self.current_query_hash = None
        self.accumulated_relevant = []
        self.accumulated_non_relevant = []
        
        # GUI Setup
        self.create_widgets()
        self.setup_layout()
        
        # Configure grid weights
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

    def create_widgets(self):
        # Control Frame
        self.control_frame = ttk.Frame(self.root)
        self.btn_search = ttk.Button(self.control_frame, text="Search", command=self.perform_search)
        self.btn_add_image = ttk.Button(self.control_frame, text="Add Image", command=self.add_image_to_db)
        self.btn_add_dir = ttk.Button(self.control_frame, text="Add Directory", command=self.add_directory_to_db)
        self.btn_delete = ttk.Button(self.control_frame, text="Delete Image", command=self.delete_image_from_db)
        self.btn_feedback = ttk.Button(self.control_frame, text="Refine Search", command=self.collect_feedback)
        
        # Results Display
        self.results_frame = ttk.LabelFrame(self.root, text="Top 10 Results")
        self.results_canvas = tk.Canvas(self.results_frame)
        self.scrollbar = ttk.Scrollbar(self.results_frame, orient="vertical", command=self.results_canvas.yview)
        self.results_content = ttk.Frame(self.results_canvas)
        
        # Query Image Display
        self.query_frame = ttk.LabelFrame(self.root, text="Query Image")
        self.query_label = ttk.Label(self.query_frame)
        
        # Status Bar
        self.status_bar = ttk.Label(self.root, text="Ready", anchor=tk.W)
        
        # Configure canvas scrolling
        self.results_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.results_canvas.create_window((0, 0), window=self.results_content, anchor="nw")
        self.results_content.bind("<Configure>", self.on_frame_configure)

    def setup_layout(self):
        # Control Frame
        self.control_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        self.btn_search.pack(side=tk.LEFT, padx=5, pady=5)
        self.btn_add_image.pack(side=tk.LEFT, padx=5, pady=5)
        self.btn_add_dir.pack(side=tk.LEFT, padx=5, pady=5)
        self.btn_delete.pack(side=tk.LEFT, padx=5, pady=5)
        self.btn_feedback.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Results Frame (Left)
        self.results_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        self.results_canvas.pack(side=tk.LEFT, fill="both", expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill="y")
        
        # Query Image Frame (Right)
        self.query_frame.grid(row=1, column=1, padx=10, pady=5, sticky="nsew")
        self.query_label.pack(padx=10, pady=150)
        
        # Status Bar
        self.status_bar.grid(row=2, column=0, columnspan=2, sticky="ew")

    def load_query_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            with open(file_path, 'rb') as f:
                file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                self.query_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if self.query_image is not None:
                self.display_image(self.query_image, self.query_label)
                self.current_query = os.path.relpath(file_path)
                self.status_bar.config(text=f"Loaded query image: {os.path.basename(file_path)}")
            else:
                self.status_bar.config(text="Failed to load image. Please check the file path.")

    def perform_search(self):
        self.load_query_image()
        if not hasattr(self, 'query_image'):
            self.status_bar.config(text="Please select a query image first!")
            return
        
        # NEW: Reset accumulated feedback if query changes
        current_hash = self._get_image_hash(self.query_image)
        if current_hash != self.current_query_hash:
            self.accumulated_relevant = []
            self.accumulated_non_relevant = []
            self.current_query_hash = current_hash
        
        # Existing code continues...
        for widget in self.results_content.winfo_children():
            widget.destroy()
        
        self.status_bar.config(text="Searching...")
        self.root.update_idletasks()
        
        try:
            results = self.db.search(self.query_image, self.feature_extractor, k=10)
            self.display_results(results)
            self.status_bar.config(text=f"Found {len(results)} results")
        except Exception as e:
            self.status_bar.config(text=f"Error: {str(e)}")

    def display_image(self, cv_image, label_widget):
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(image)
        
        label_widget.configure(image=photo)
        label_widget.image = photo
    
    def display_results(self, results):
        # Clear previous grid configuration
        for widget in self.results_content.winfo_children():
            widget.destroy()

        # Configure grid for 2 columns
        self.results_content.grid_columnconfigure(0, weight=1)
        self.results_content.grid_columnconfigure(1, weight=1)

        # MODIFIED: Initialize both feedback lists
        self.relevant_images = []
        self.non_relevant_images = []

        for i, (path, score) in enumerate(results):
            # Calculate row and column positions
            row = i // 2
            col = i % 2

            frame = ttk.Frame(self.results_content)
            frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")

            # Load and display image
            img = cv2.imread(path)
            if img is not None:
                image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image.thumbnail((150, 150))
                photo = ImageTk.PhotoImage(image)

                label = ttk.Label(frame, image=photo)
                label.image = photo
                label.pack()

                # Add score and filename labels
                ttk.Label(frame, text=f"Score: {score:.4f}").pack()
                ttk.Label(frame, text=os.path.basename(path)).pack()

                # MODIFIED: Dual checkboxes
                feedback_frame = ttk.Frame(frame)
                feedback_frame.pack(pady=5)
                
                rel_var = tk.BooleanVar()
                non_rel_var = tk.BooleanVar()
                
                ttk.Checkbutton(feedback_frame, text="Relevant", variable=rel_var).pack(side=tk.LEFT)
                ttk.Checkbutton(feedback_frame, text="Non-relevant", variable=non_rel_var).pack(side=tk.LEFT)
                
                self.relevant_images.append((path, rel_var))
                self.non_relevant_images.append((path, non_rel_var))

            # Configure row growth
            self.results_content.grid_rowconfigure(row, weight=1)
    
    def collect_feedback(self):
        relevant_paths = [path for path, var in self.relevant_images if var.get()]
        non_relevant_paths = [path for path, var in self.non_relevant_images if var.get()]
        
        if not relevant_paths and not non_relevant_paths:
            messagebox.showinfo("Info", "No feedback provided.")
            return

        # MODIFIED: Extract and accumulate features
        new_relevant = []
        new_non_relevant = []
        
        for path in relevant_paths:
            img = cv2.imread(path)
            features = self.feature_extractor.extract_features(img)
            new_relevant.append(features)
            
        for path in non_relevant_paths:
            img = cv2.imread(path)
            features = self.feature_extractor.extract_features(img)
            new_non_relevant.append(features)
        
        # Add to accumulated vectors
        self.accumulated_relevant.extend(new_relevant)
        self.accumulated_non_relevant.extend(new_non_relevant)
        
        # Pass all accumulated features to refinement
        self.refine_query(self.accumulated_relevant, self.accumulated_non_relevant)
    
    def refine_query(self, relevant_features, non_relevant_features):
        # Rocchio algorithm parameters
        alpha = 0.6  # Weight for original query
        beta = 0.8   # Weight for relevant images
        gamma = 0.8  # Weight for non-relevant images
        top_k = 10

        # Extract and normalize original query features
        original_features = self.feature_extractor.extract_features(self.query_image)
        original_features = original_features / np.linalg.norm(original_features)

        # Process accumulated relevant features
        avg_relevant = np.zeros_like(original_features)
        if len(relevant_features) > 0:
            relevant_array = np.array(relevant_features)
            relevant_norms = np.linalg.norm(relevant_array, axis=1, keepdims=True)
            normalized_relevant = relevant_array / np.clip(relevant_norms, 1e-8, None)
            avg_relevant = np.mean(normalized_relevant, axis=0)
            if np.linalg.norm(avg_relevant) > 0:
                avg_relevant /= np.linalg.norm(avg_relevant)

        # Process accumulated non-relevant features
        avg_non_relevant = np.zeros_like(original_features)
        if len(non_relevant_features) > 0:
            non_rel_array = np.array(non_relevant_features)
            non_rel_norms = np.linalg.norm(non_rel_array, axis=1, keepdims=True)
            normalized_non_rel = non_rel_array / np.clip(non_rel_norms, 1e-8, None)
            avg_non_relevant = np.mean(normalized_non_rel, axis=0)
            if np.linalg.norm(avg_non_relevant) > 0:
                avg_non_relevant /= np.linalg.norm(avg_non_relevant)

        # Apply Rocchio formula with accumulated feedback
        refined_query = alpha * original_features + beta * avg_relevant - gamma * avg_non_relevant
        refined_query /= np.linalg.norm(refined_query)

        # Search with refined query
        distances, indices = self.db.index.search(
            np.array([refined_query]).astype('float32'), k=top_k
        )

        # Process results with proper similarity conversion
        cursor = self.db.conn.cursor()
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < 0:  # Handle FAISS invalid indices
                continue
                
            cursor.execute("SELECT path, features FROM images WHERE rowid=?", (int(idx)+1,))
            row = cursor.fetchone()
            if row:
                path = row[0]
                normalized_score = (score + 1) / 2
                results.append((path, float(normalized_score)))

        self.display_results(results)
        self.status_bar.config(text=f"Refined search (Relevant: {len(self.accumulated_relevant)}, Non-relevant: {len(self.accumulated_non_relevant)}) - {len(results)} results")
    
    def _get_image_hash(self, image):
        return hashlib.md5(image.tobytes()).hexdigest()

    def add_image_to_db(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            relative_path = os.path.relpath(file_path)
            self.db.add_image(relative_path, self.feature_extractor)
            messagebox.showinfo("Success", "Image added to database.")
    
    def add_directory_to_db(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            for dirpath, dirnames, _ in os.walk(dir_path):
                if not dirnames:  # Check if there are no subdirectories
                    self.db.add_directory(os.path.relpath(dirpath), feature_extractor=self.feature_extractor, label="nothing")
            messagebox.showinfo("Success", "Directory added to database.")
    
    def delete_image_from_db(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.db.delete_image(os.path.relpath(file_path))
            messagebox.showinfo("Success", "Image deleted from database.")
    
    def on_frame_configure(self, _):
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("900x700")
    app = CBIRApp(root)
    root.mainloop()