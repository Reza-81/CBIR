import numpy as np
from rembg import remove
from PIL import Image
import io
import cv2
from skimage import color, measure
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops
from skimage.color import rgb2gray
import pywt
from scipy.fftpack import dct


def remove_background(image_array: np.ndarray) -> np.ndarray:
    """
    Removes the background from an image using rembg and ensures the output is in RGB format.

    Args:
        image_array (np.ndarray): The input image as a NumPy array (H, W, C).

    Returns:
        np.ndarray: The image with the background removed, in RGB format (H, W, 3).
    """
    # Convert the NumPy array to a PIL Image
    pil_image = Image.fromarray(image_array)
    
    # Convert the PIL Image to bytes for rembg processing
    image_bytes = io.BytesIO()
    pil_image.save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()
    
    # Process the image with rembg
    output_bytes = remove(image_bytes)
    
    # Convert the processed bytes back to a PIL Image
    output_image = Image.open(io.BytesIO(output_bytes))
    
    # Ensure the output is in RGB format (drops alpha channel if present)
    output_image = output_image.convert("RGB")
    
    # Convert the PIL Image back to a NumPy array
    return np.array(output_image)


def major_color(segmented_image, k=1):
    # Ensure the image is in the correct format (RGB or BGR)
    if segmented_image.ndim == 3 and segmented_image.shape[2] == 3:
        # Mask out the background (assuming background is zero)
        mask = segmented_image[:, :, 0] > 0  # Segment is assumed to be non-zero

        # Extract the non-zero (segmented) region pixels
        pixels = segmented_image[mask]

        # If there are pixels in the segment, apply K-means
        if pixels.size > 0:
            # Reshape pixels array for K-means
            pixels = pixels.reshape(-1, 3).astype(np.float32)

            # Apply K-means to find the most dominant color
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            # The most dominant color is the center of the largest cluster
            dominant_color = centers.astype(int)
            return dominant_color
        else:
            return None  # No segmented pixels found
    else:
        return None  # Image is not in RGB/BGR format
    

def calculate_circularity(segmented_image, feature_length=5):
    # Convert to grayscale if the image has multiple channels (e.g., RGB)
    if segmented_image.ndim > 2:
        segmented_image = rgb2gray(segmented_image)
    
    # Ensure binary format: threshold at a small value to create binary image
    segmented_image = segmented_image > 0
    
    # Label the segmented regions
    labeled_image = label(segmented_image)
    
    # Calculate properties for each labeled region
    circularities = []
    for region in regionprops(labeled_image):
        if region.perimeter > 0:  # Avoid division by zero
            circularity = (4 * np.pi * region.area) / (region.perimeter ** 2)
            circularities.append(circularity)
    
    # Compute statistics for fixed-size feature vector
    if len(circularities) > 0:
        circularities = np.array(circularities)
        feature_vector = np.array([
            np.mean(circularities),
            np.std(circularities),
            np.min(circularities),
            np.max(circularities),
            np.median(circularities)
        ])
    else:
        feature_vector = np.zeros(feature_length)
    
    return feature_vector


def calculate_shape_descriptors(segmented_image):
    # Convert to grayscale if the image has multiple channels (e.g., RGB)
    if segmented_image.ndim > 2:
        segmented_image = rgb2gray(segmented_image)
    
    # Ensure binary format: threshold at a small value to create binary image
    segmented_image = segmented_image > 0
    
    labeled_image = measure.label(segmented_image)
    props = measure.regionprops(labeled_image)[0]  # Assuming one object in the segmentation
    
    eccentricity = props.eccentricity
    solidity = props.solidity
    aspect_ratio = props.bbox[3] / props.bbox[2]  # Width / Height of bounding box
    
    return eccentricity, solidity, aspect_ratio


def compute_hog_with_opencv(image, fixed_size=(128, 64), visualize=False):
    """
    Compute the HOG (Histogram of Oriented Gradients) features of an image with a fixed size.

    Args:
        image (np.ndarray): Input image (RGB or Grayscale).
        fixed_size (tuple): Desired fixed size for the input image (width, height).
        visualize (bool): Whether to return a HOG visualization.

    Returns:
        features (np.ndarray): The HOG descriptor feature vector.
        hog_image (np.ndarray): A visualization of the HOG (optional).
    """
    # Resize the image to a fixed size
    resized_image = cv2.resize(image, fixed_size, interpolation=cv2.INTER_AREA)

    # Convert to grayscale if needed
    if len(resized_image.shape) == 3:
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = resized_image

    # Set up the HOGDescriptor
    hog = cv2.HOGDescriptor(
        _winSize=(fixed_size[0], fixed_size[1]),  # Use the fixed size for window
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )

    # Compute the HOG descriptor
    features = hog.compute(gray_image)

    if visualize:
        # Visualize the HOG image (optional)
        gx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=1)
        magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        hog_image = np.zeros_like(gray_image, dtype=np.float32)

        cell_size = 8
        for i in range(0, gray_image.shape[0] // cell_size):
            for j in range(0, gray_image.shape[1] // cell_size):
                cell_mag = magnitude[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
                cell_angle = angle[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
                hist, _ = np.histogram(cell_angle, bins=9, range=(0, 180), weights=cell_mag)
                hog_image[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size] = np.sum(hist)

        hog_image = cv2.normalize(hog_image, None, 0, 255, cv2.NORM_MINMAX)

        return features.flatten(), hog_image.astype(np.uint8)
    return features.flatten()


def compute_harris_features(image: np.ndarray, k: float = 0.04, percentile: float = 99.5) -> np.ndarray:
    """
    Compute Harris corner features for an image with an adaptive threshold.

    Args:
        image (np.ndarray): Input image as a NumPy array (grayscale or RGB).
        k (float): Harris detector free parameter (typically 0.04 to 0.06).
        percentile (float): Percentile for adaptive thresholding (e.g., 99.5 for top 0.5% of responses).

    Returns:
        np.ndarray: A binary mask highlighting detected corners.
    """
    # Convert image to grayscale if it's not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Convert to float32 for precision in calculations
    gray = np.float32(gray)

    # Apply the Harris corner detection
    harris_response = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=k)

    # Normalize the response for better visualization
    harris_response = cv2.normalize(harris_response, None, 0, 255, norm_type=cv2.NORM_MINMAX)

    # Compute the adaptive threshold
    threshold_value = np.percentile(harris_response, percentile)

    # Thresholding to identify strong corners
    corners = harris_response > threshold_value

    return corners.astype(np.uint8) * 255, harris_response


def compute_sift_features(image: np.ndarray) -> tuple:
    """
    Compute the SIFT (Scale-Invariant Feature Transform) keypoints and descriptors for an image.

    Args:
        image (np.ndarray): Input image as a NumPy array (grayscale or RGB).

    Returns:
        tuple: A tuple containing:
            - keypoints (List of cv2.KeyPoint): Detected keypoints.
            - descriptors (np.ndarray): SIFT descriptors for each keypoint.
    """
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Initialize the SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    return keypoints, descriptors


def compute_glcm_features(image: np.ndarray, distances: list = [1], angles: list = [0], levels: int = 256) -> dict:
    """
    Compute GLCM (Gray Level Co-occurrence Matrix) features for an image.

    Args:
        image (np.ndarray): Input image as a NumPy array (grayscale or RGB).
        distances (list): List of distances to compute the co-occurrence matrix.
        angles (list): List of angles to compute the co-occurrence matrix (in radians).
        levels (int): Number of gray levels in the image. Default is 256.

    Returns:
        dict: A dictionary of GLCM features, including 'contrast', 'correlation', 'energy', and 'homogeneity'.
    """
    # Convert image to grayscale if it's not already
    if len(image.shape) == 3:
        gray_image = color.rgb2gray(image)
        gray_image = (gray_image * (levels - 1)).astype(np.uint8)  # Normalize to [0, levels-1]
    else:
        gray_image = image

    # Compute the GLCM (Gray Level Co-occurrence Matrix)
    glcm = graycomatrix(gray_image, distances=distances, angles=angles, symmetric=True, normed=True)

    # Extract GLCM properties
    glcm_features = {
        'contrast': graycoprops(glcm, prop='contrast'),
        'correlation': graycoprops(glcm, prop='correlation'),
        'energy': graycoprops(glcm, prop='energy'),
        'homogeneity': graycoprops(glcm, prop='homogeneity')
    }

    # Average the values over the different angles (if multiple angles are used)
    glcm_features = {k: np.mean(v) for k, v in glcm_features.items()}

    return glcm_features


def compute_gist_like_features(image: np.ndarray, fixed_size=(256, 256), 
                               num_scales=4, num_orientations=8, grid_size=(4, 4)) -> np.ndarray:
    """
    Compute a GIST-like descriptor for an image using Gabor filters.

    Args:
        image (np.ndarray): Input image (RGB or Grayscale).
        num_scales (int): Number of scales for Gabor filters.
        num_orientations (int): Number of orientations for Gabor filters.
        grid_size (tuple): Number of divisions for spatial grid (rows, cols).

    Returns:
        np.ndarray: GIST-like feature vector.
    """
    # Resize the image to a fixed size
    resized_image = cv2.resize(image, fixed_size, interpolation=cv2.INTER_AREA)

    # Convert to grayscale
    if len(resized_image.shape) == 3:
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = resized_image

    # Define Gabor filter bank
    gabor_kernels = []
    for scale in range(num_scales):
        for orientation in range(num_orientations):
            theta = orientation * np.pi / num_orientations
            kernel = cv2.getGaborKernel((15, 15), sigma=2.5 * (2 ** scale), theta=theta, lambd=10, gamma=0.5, psi=0)
            gabor_kernels.append(kernel)

    # Apply Gabor filters and compute energy
    energy_responses = []
    for kernel in gabor_kernels:
        filtered_image = cv2.filter2D(gray_image, cv2.CV_32F, kernel)
        energy = np.square(filtered_image)
        energy_responses.append(energy)

    # Divide image into a grid and compute average energy per block
    rows, cols = grid_size
    gist_like_features = []
    for energy in energy_responses:
        h, w = energy.shape
        grid_h, grid_w = h // rows, w // cols
        for i in range(rows):
            for j in range(cols):
                block = energy[i * grid_h:(i + 1) * grid_h, j * grid_w:(j + 1) * grid_w]
                gist_like_features.append(np.mean(block))

    return np.array(gist_like_features)


def extract_fourier_feature_vector(image: np.ndarray, vector_size: int = 128) -> np.ndarray:
    """
    Extracts a compact Fourier feature vector from an image.

    Args:
        image (np.ndarray): Input image as a NumPy array (grayscale or RGB).
        vector_size (int): Size of the feature vector to reduce the Fourier magnitude spectrum.

    Returns:
        np.ndarray: A 1D feature vector of size (vector_size,).
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image

    # Compute the 2D FFT and shift zero frequency to center
    f_transform = np.fft.fft2(gray_image)
    f_shifted = np.fft.fftshift(f_transform)

    # Compute the magnitude spectrum
    magnitude = np.abs(f_shifted)
    log_magnitude = np.log(1 + magnitude)  # Log scale for better representation

    # Resize the magnitude spectrum to a fixed size (compact representation)
    resized_spectrum = cv2.resize(log_magnitude, (vector_size, 1), interpolation=cv2.INTER_AREA)

    # Flatten the feature vector
    feature_vector = resized_spectrum.flatten()

    return feature_vector


def compute_wavelet_features(image: np.ndarray, wavelet: str = 'haar', level: int = 3, feature_length: int = 128) -> np.ndarray:
    """
    Compute Wavelet features for an image and return a fixed-length feature vector.

    Args:
        image (np.ndarray): Input image (grayscale or RGB).
        wavelet (str): Type of wavelet to use (e.g., 'haar', 'db1', 'sym2', etc.).
        level (int): Level of wavelet decomposition.
        feature_length (int): Desired length of the output feature vector.

    Returns:
        np.ndarray: Fixed-length Wavelet feature vector.
    """
    # Step 1: Convert to grayscale if necessary
    if len(image.shape) == 3:  # RGB image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Step 2: Resize the image to ensure consistency
    fixed_size = (256, 256)
    image = cv2.resize(image, fixed_size, interpolation=cv2.INTER_AREA)

    # Step 3: Perform wavelet decomposition
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)

    # Flatten all wavelet coefficients
    features = []
    for i in range(len(coeffs)):
        if i == 0:  # Approximation coefficients at the current level
            cA = coeffs[i]
            features.append(np.mean(cA))
            features.append(np.var(cA))
            features.append(np.sum(np.abs(cA)))  # Energy
        else:  # Horizontal, vertical, and diagonal coefficients
            cH, cV, cD = coeffs[i]
            features.extend([np.mean(cH), np.var(cH), np.sum(np.abs(cH))])
            features.extend([np.mean(cV), np.var(cV), np.sum(np.abs(cV))])
            features.extend([np.mean(cD), np.var(cD), np.sum(np.abs(cD))])

    # Step 4: Pad or truncate the feature vector to the desired length
    feature_vector = np.array(features)
    if len(feature_vector) > feature_length:
        feature_vector = feature_vector[:feature_length]  # Truncate
    else:
        padding = feature_length - len(feature_vector)
        feature_vector = np.pad(feature_vector, (0, padding), mode='constant')

    return feature_vector


def compute_dct_features(image: np.ndarray, feature_length: int = 128) -> np.ndarray:
    """
    Compute DCT features for an image and return a fixed-length feature vector.

    Args:
        image (np.ndarray): Input image (grayscale or RGB).
        feature_length (int): Desired length of the output feature vector.

    Returns:
        np.ndarray: Fixed-length DCT feature vector.
    """
    # Step 1: Convert to grayscale if necessary
    if len(image.shape) == 3:  # RGB image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Step 2: Resize the image to ensure consistency
    fixed_size = (256, 256)  # Resize to fixed size (adjust as necessary)
    image = cv2.resize(image, fixed_size, interpolation=cv2.INTER_AREA)

    # Step 3: Apply 2D DCT (Discrete Cosine Transform)
    # DCT is applied on both rows and columns
    dct_image = dct(dct(image.T, norm='ortho').T, norm='ortho')

    # Step 4: Flatten the DCT coefficients (after DCT, we have a 2D array)
    dct_flat = dct_image.flatten()

    # Step 5: Select the top coefficients (lower-frequency components)
    # Sort by magnitude (absolute value) and pick the most important coefficients
    sorted_indices = np.argsort(np.abs(dct_flat))  # Sorting by absolute value of DCT coefficients
    top_indices = sorted_indices[:feature_length]  # Select top 'feature_length' indices

    # Step 6: Create the feature vector using selected coefficients
    feature_vector = np.zeros(feature_length)
    feature_vector[:len(top_indices)] = dct_flat[top_indices]

    return feature_vector


def fourier_boundary_descriptor(image: np.ndarray, num_descriptors: int = 128) -> np.ndarray:
    """
    Extract boundary descriptor using Fourier transform on image contours.
    
    Args:
        image (np.ndarray): Input image (RGB or Grayscale)
        num_descriptors (int): Desired length of output feature vector
        
    Returns:
        np.ndarray: Fourier boundary descriptor vector of shape (num_descriptors,)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Threshold to get binary image
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        return np.zeros(num_descriptors)
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Convert contour to complex numbers (x + iy)
    contour_complex = np.empty(len(largest_contour), dtype=complex)
    for i, point in enumerate(largest_contour):
        contour_complex[i] = complex(point[0][0], point[0][1])
    
    # Apply Fourier Transform
    fourier_result = np.fft.fft(contour_complex)
    
    # Keep low-frequency components (excluding DC component)
    coefficients = fourier_result[1:num_descriptors+1]
    
    # Get magnitude spectrum
    magnitudes = np.abs(coefficients)
    
    # Normalize to fixed size
    if len(magnitudes) < num_descriptors:
        # Pad with zeros if needed
        magnitudes = np.pad(magnitudes, (0, num_descriptors - len(magnitudes)), 'constant')
    else:
        # Truncate to desired length
        magnitudes = magnitudes[:num_descriptors]
    
    return magnitudes