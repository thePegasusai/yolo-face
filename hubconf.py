# hubconf.py
# This file defines entry points for loading yolo-face models via torch.hub
# Place this file in the root directory of your cloned yolo-face repository.

dependencies = ['torch', 'ultralytics']  # Specify necessary dependencies

import os
import torch
from ultralytics import YOLO # Import the YOLO class from ultralytics

def _load_face_model(model_filename, pretrained=True, *args, **kwargs):
    """
    Internal helper function to load a specific YOLO face model using the
    ultralytics package.

    Args:
        model_filename (str): The name of the .pt model file (e.g., 'yolov11n-face.pt').
        pretrained (bool): If True, attempts to load the specified model weights.
                           In this setup, it ensures the file exists locally.
    """
    # Construct the full path to the model file relative to this hubconf.py file
    # __file__ gives the path to hubconf.py itself
    hub_dir = os.path.dirname(__file__)
    model_path = os.path.join(hub_dir, model_filename)

    if not os.path.exists(model_path):
        # If loading directly from GitHub in the future, this error message might
        # need adjustment, as the file might need to be downloaded first.
        # For local loading, this check is essential.
        raise FileNotFoundError(f"Model file '{model_filename}' not found in the repository root: {hub_dir}. "
                                f"Ensure the .pt file exists in the same directory as hubconf.py.")

    if not pretrained:
        # You could potentially load just the model architecture without weights here
        # if the ultralytics YOLO class supports it, but usually, we want the weights.
        print(f"Warning: Loading {model_filename} with pretrained=False is not standard for this hubconf. "
              f"Attempting to load structure only (may fail).")
        # model = YOLO(model_filename).model # Example, syntax might differ
        # return model
        raise ValueError("Loading without pretrained weights is not fully supported by this hubconf setup.")


    try:
        # Use the ultralytics YOLO class to load the model from the .pt file
        # This assumes the .pt file contains both architecture and weights.
        print(f"Loading model from local path: {model_path}")
        model = YOLO(model_path)
        print(f"Successfully loaded {model_filename} using ultralytics.")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_filename}' using ultralytics from path {model_path}: {e}")

# --- Define Entry Points for Specific Models ---
# Each function corresponds to a model name you can pass to torch.hub.load

def yolov11n_face(pretrained=True, *args, **kwargs):
    """Loads the YOLOv11n-face model."""
    return _load_face_model('yolov11n-face.pt', pretrained=pretrained, *args, **kwargs)

def yolov11s_face(pretrained=True, *args, **kwargs):
    """Loads the YOLOv11s-face model."""
    return _load_face_model('yolov11s-face.pt', pretrained=pretrained, *args, **kwargs)

def yolov11m_face(pretrained=True, *args, **kwargs):
    """Loads the YOLOv11m-face model."""
    return _load_face_model('yolov11m-face.pt', pretrained=pretrained, *args, **kwargs)

def yolov11l_face(pretrained=True, *args, **kwargs):
    """Loads the YOLOv11l-face model."""
    return _load_face_model('yolov11l-face.pt', pretrained=pretrained, *args, **kwargs)

def yolov10n_face(pretrained=True, *args, **kwargs):
    """Loads the YOLOv10n-face model."""
    return _load_face_model('yolov10n-face.pt', pretrained=pretrained, *args, **kwargs)

def yolov10s_face(pretrained=True, *args, **kwargs):
    """Loads the YOLOv10s-face model."""
    return _load_face_model('yolov10s-face.pt', pretrained=pretrained, *args, **kwargs)

def yolov10m_face(pretrained=True, *args, **kwargs):
    """Loads the YOLOv10m-face model."""
    return _load_face_model('yolov10m-face.pt', pretrained=pretrained, *args, **kwargs)

def yolov10l_face(pretrained=True, *args, **kwargs):
    """Loads the YOLOv10l-face model."""
    return _load_face_model('yolov10l-face.pt', pretrained=pretrained, *args, **kwargs)

def yolov8n_face(pretrained=True, *args, **kwargs):
    """Loads the YOLOv8n-face model."""
    return _load_face_model('yolov8n-face.pt', pretrained=pretrained, *args, **kwargs)

def yolov8m_face(pretrained=True, *args, **kwargs):
    """Loads the YOLOv8m-face model."""
    return _load_face_model('yolov8m-face.pt', pretrained=pretrained, *args, **kwargs)

def yolov8l_face(pretrained=True, *args, **kwargs):
    """Loads the YOLOv8l-face model."""
    return _load_face_model('yolov8l-face.pt', pretrained=pretrained, *args, **kwargs)

def yolov6n_face(pretrained=True, *args, **kwargs):
    """Loads the YOLOv6n-face model."""
    return _load_face_model('yolov6n-face.pt', pretrained=pretrained, *args, **kwargs)

def yolov6m_face(pretrained=True, *args, **kwargs):
    """Loads the YOLOv6m-face model."""
    return _load_face_model('yolov6m-face.pt', pretrained=pretrained, *args, **kwargs)

# Add other models from the repository here if needed following the same pattern
# Example for a person model (if you download yolov8n-person.pt)
# def yolov8n_person(pretrained=True, *args, **kwargs):
#     """Loads the YOLOv8n-person model."""
#     return _load_face_model('yolov8n-person.pt', pretrained=pretrained, *args, **kwargs)