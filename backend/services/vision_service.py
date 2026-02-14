import os
from datetime import datetime

"""
Minimal Vision service replacement.
This module no longer runs Vision analysis directly. It only registers uploaded images
and returns simple metadata so the frontend can proceed to send
the image into the chat flow.
"""

UPLOAD_DIR = os.path.join("uploads", "images")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def register_uploaded_image(stored_path: str, original_filename: str):
    """Record minimal metadata for an uploaded image.

    Args:
        stored_path: Full path where the file was saved.
        original_filename: Original filename from the client.

    Returns:
        dict with basic metadata (id placeholder, filename, stored_path, created_at)
    """
    created_at = datetime.utcnow()
    # Construct a public URL for the uploaded file (served by FastAPI static mount)
    public_path = f"/uploads/images/{os.path.basename(stored_path)}"
    return {
        "id": -1,
        "filename": original_filename,
        "stored_path": stored_path,
        "image_url": public_path,
        "created_at": created_at,
    }
