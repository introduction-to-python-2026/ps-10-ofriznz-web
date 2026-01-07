from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    import numpy as np
from PIL import Image

def load_image(image_path):

    try:
        img = Image.open(image_path)
        return np.array(img)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading the image: {e}")
        return None

# Test the function
image_path = '/content/Waterfall.jpg'
loaded_image = load_image(image_path)

if loaded_image is not None:
    print(f"Image loaded successfully. Shape: {loaded_image.shape}")
    print(f"Data type: {loaded_image.dtype}")
else:
    print("Failed to load image.")

def edge_detection(image):
    import numpy as np
from scipy.signal import convolve2d

def edge_detection(image_array):
    
    # Convert to grayscale
    # Check if the image is already grayscale (2D array) or has an alpha channel (4D array)
    if image_array.ndim == 2:
        grayscale_image = image_array
    elif image_array.shape[2] == 4: # RGBA image
        # Average R, G, B channels, ignoring A
        grayscale_image = np.mean(image_array[:, :, :3], axis=2)
    else: # RGB image
        grayscale_image = np.mean(image_array, axis=2)

    # Define kernels
    kernelY = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])
    kernelX = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    # Apply convolution
    edgeY = convolve2d(grayscale_image, kernelY, mode='same', boundary='fill', fillvalue=0.0)
    edgeX = convolve2d(grayscale_image, kernelX, mode='same', boundary='fill', fillvalue=0.0)

    # Compute edge magnitude
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

    return edgeMAG

# Test the function with the loaded image
if 'loaded_image' in locals():
    edge_magnitude = edge_detection(loaded_image)
    print(f"Edge magnitude array shape: {edge_magnitude.shape}")
    print(f"Edge magnitude array data type: {edge_magnitude.dtype}")
else:
    print("Please load an image first using the load_image function.")
