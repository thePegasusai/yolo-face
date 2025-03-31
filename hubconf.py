# hubconf.py (UPDATED FOR GITHUB RELEASE DOWNLOAD)
# This file defines entry points for loading yolo-face models via torch.hub
# It now includes logic to download .pt files from GitHub Releases if not cached.

dependencies = ['torch', 'ultralytics', 'requests'] # Added requests for robust download check

import os
import torch
import requests # Needed for checking URL status before download (optional but good)
import torch.hub
from ultralytics import YOLO # Import the YOLO class from ultralytics

# --- Define Model URLs ---
# Map model filenames to their direct download URLs from GitHub Releases
# You MUST get the correct URL for each asset you upload.
# Go to your release page, right-click the asset link (e.g., yolov11s-face.pt)
# and select "Copy Link Address".
MODEL_URLS = {
    # --- REPLACE WITH YOUR ACTUAL URLs ---
    'yolov11s-face.pt': 'https://github.com/thePegasusai/yolo-face/releases/download/yolo11facemodelfile/yolov11s-face.pt',
    # 'yolov11n-face.pt': 'https://github.com/thePegasusai/yolo-face/releases/download/YOUR_TAG_v11n/yolov11n-face.pt',
    # 'yolov11m-face.pt': 'https://github.com/thePegasusai/yolo-face/releases/download/YOUR_TAG_v11m/yolov11m-face.pt',
    # 'yolov11l-face.pt': 'https://github.com/thePegasusai/yolo-face/releases/download/YOUR_TAG_v11l/yolov11l-face.pt',
    # 'yolov10n-face.pt': 'https://github.com/thePegasusai/yolo-face/releases/download/YOUR_TAG_v10n/yolov10n-face.pt',
    # --- Add other model URLs as you create releases for them ---
}

def _check_url_exists(url):
    """Checks if a URL is accessible without downloading the whole file."""
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        # Consider status codes 200 OK and 3xx redirects as valid
        return response.status_code >= 200 and response.status_code < 400
    except requests.exceptions.RequestException as e:
        print(f"Warning: Could not check URL {url}: {e}")
        return False # Assume it might exist if check fails

def _download_if_needed(model_filename, progress=True):
    """Downloads the model file from GitHub Releases if not already cached by torch.hub."""
    if model_filename not in MODEL_URLS:
        raise KeyError(f"Model '{model_filename}' URL not defined in hubconf.py MODEL_URLS. "
                       f"Available models: {list(MODEL_URLS.keys())}")

    url = MODEL_URLS[model_filename]

    # Use torch.hub's standard cache directory mechanism
    hub_dir = torch.hub.get_dir()
    # Usually weights are stored in 'checkpoints' subdirectory within hub cache
    model_dir = os.path.join(hub_dir, 'checkpoints')
    os.makedirs(model_dir, exist_ok=True)
    # The final expected path in the cache
    cached_file_path = os.path.join(model_dir, model_filename)

    # Check if the file already exists in the cache
    if not os.path.exists(cached_file_path):
        print(f"'{model_filename}' not found in cache ({model_dir}).")
        print(f"Attempting to download from: {url}")

        # Optional: Check if URL is valid before attempting download
        if not _check_url_exists(url):
             raise ConnectionError(f"URL for {model_filename} seems invalid or unreachable: {url}")

        try:
            # Use torch.hub's download utility, which handles caching etc.
            # It will download to a temp location and move to cached_file_path
            torch.hub.download_url_to_file(url, cached_file_path, progress=progress)
            print(f"Download complete. Model cached at: {cached_file_path}")
        except Exception as e:
            # Attempt to clean up partially downloaded file if error occurs
            if os.path.exists(cached_file_path):
                 try:
                     os.remove(cached_file_path)
                     print(f"Removed partially downloaded file: {cached_file_path}")
                 except OSError as rm_e:
                     print(f"Warning: Could not remove partially downloaded file {cached_file_path}: {rm_e}")
            raise RuntimeError(f"Failed to download {model_filename} from {url}: {e}")
    else:
        print(f"Using cached model: {cached_file_path}")

    # --- Optional Sanity Check ---
    try:
        # Check if file has size > 0
        if os.path.getsize(cached_file_path) == 0:
             raise OSError(f"Cached file {cached_file_path} is empty. "
                           f"Cache might be corrupt. Please delete it manually and retry.")
        # You could add more checks here (e.g., hash check if available)
    except Exception as e:
         # Don't automatically delete here, as it might be a temporary issue.
         # Let the YOLO loading attempt handle potential corruption.
         print(f"Warning: Could not properly validate cached file {cached_file_path}: {e}")
    # --- End Sanity Check ---

    return cached_file_path # Return the path to the cached file


def _load_face_model(model_filename, pretrained=True, progress=True, *args, **kwargs):
    """
    Internal helper function to load a specific YOLO face model using the
    ultralytics package, downloading from GitHub Releases if necessary.
    """
    if not pretrained:
        raise ValueError("Loading without pretrained weights is not supported by this hubconf.")

    # Step 1: Ensure the .pt file is downloaded and available locally in the cache
    local_model_path = _download_if_needed(model_filename, progress=progress)

    # Step 2: Load the model from the local file using ultralytics.YOLO
    try:
        print(f"Loading model using ultralytics.YOLO from cached path: {local_model_path}")
        # The YOLO class handles loading the architecture and weights from the .pt file
        model = YOLO(local_model_path)
        print(f"Successfully loaded {model_filename} using ultralytics.")
        return model
    except FileNotFoundError:
        # This might happen if _download_if_needed failed silently or cache got deleted between checks
         raise FileNotFoundError(f"Model file expected at {local_model_path} but not found after download attempt. "
                                 f"Check cache permissions or delete cache and retry.")
    except Exception as e:
        # Catch other potential errors during YOLO() initialization
        raise RuntimeError(f"Failed to load model '{model_filename}' using ultralytics.YOLO from path {local_model_path}: {e}")

# --- Define Entry Points for Specific Models ---
# These functions now trigger the download logic via _load_face_model
# The 'pretrained' argument ensures download happens.
# The 'progress' argument controls the download progress bar visibility.

def yolov11s_face(pretrained=True, progress=True, *args, **kwargs):
    """Loads the YOLOv11s-face model, downloading from GitHub Releases if needed."""
    return _load_face_model('yolov11s-face.pt', pretrained=pretrained, progress=progress, *args, **kwargs)

# --- Add functions for other models below ---
# Make sure the function name matches the 'model' argument in torch.hub.load
# and the filename matches the key in MODEL_URLS and the actual .pt file name.

# Example for yolov11n:
# def yolov11n_face(pretrained=True, progress=True, *args, **kwargs):
#     """Loads the YOLOv11n-face model, downloading from GitHub Releases if needed."""
#     # Ensure 'yolov11n-face.pt' key exists with a valid URL in MODEL_URLS above
#     return _load_face_model('yolov11n-face.pt', pretrained=pretrained, progress=progress, *args, **kwargs)

# Example for yolov8n:
# def yolov8n_face(pretrained=True, progress=True, *args, **kwargs):
#     """Loads the YOLOv8n-face model, downloading from GitHub Releases if needed."""
#     # Ensure 'yolov8n-face.pt' key exists with a valid URL in MODEL_URLS above
#     return _load_face_model('yolov8n-face.pt', pretrained=pretrained, progress=progress, *args, **kwargs)

# ... add other functions as needed ...