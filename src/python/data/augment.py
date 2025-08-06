"""
Image augmentation tools for terrain prediction dataset
"""

import os
import numpy as np
from pathlib import Path
from typing import Union, Tuple
from PIL import Image, ImageOps

import logging


def rotate_image(image: Union[np.ndarray, Image.Image], angle: float) -> np.ndarray:
    """
    Rotate an image by a given angle in degrees.
    
    Args:
        image: Input image as numpy array (H, W) or (H, W, C) or PIL Image
        angle: Rotation angle in degrees (positive = counterclockwise)
        
    Returns:
        Rotated image as numpy array
    """
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            # Grayscale
            pil_image = Image.fromarray(image, mode='L')
        elif len(image.shape) == 3:
            if image.shape[2] == 3:
                pil_image = Image.fromarray(image, mode='RGB')
            elif image.shape[2] == 4:
                pil_image = Image.fromarray(image, mode='RGBA')
            else:
                # Single channel but 3D array
                pil_image = Image.fromarray(image.squeeze(), mode='L')
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
    else:
        pil_image = image
    
    # Rotate the image
    rotated_pil = pil_image.rotate(-angle, expand=False, fillcolor=0)
    
    # Convert back to numpy array
    rotated = np.array(rotated_pil)
    
    return rotated


def flip_image(image: Union[np.ndarray, Image.Image], flip_code: int = 1) -> np.ndarray:
    """
    Flip an image horizontally, vertically, or both.
    
    Args:
        image: Input image as numpy array (H, W) or (H, W, C) or PIL Image
        flip_code: 0 = vertical flip, 1 = horizontal flip, -1 = both
        
    Returns:
        Flipped image as numpy array
    """
    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            # Grayscale
            pil_image = Image.fromarray(image, mode='L')
        elif len(image.shape) == 3:
            if image.shape[2] == 3:
                pil_image = Image.fromarray(image, mode='RGB')
            elif image.shape[2] == 4:
                pil_image = Image.fromarray(image, mode='RGBA')
            else:
                # Single channel but 3D array
                pil_image = Image.fromarray(image.squeeze(), mode='L')
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
    else:
        pil_image = image
    
    # Apply flip operations
    if flip_code == 1:  # Horizontal flip
        flipped_pil = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
    elif flip_code == 0:  # Vertical flip
        flipped_pil = pil_image.transpose(Image.FLIP_TOP_BOTTOM)
    elif flip_code == -1:  # Both horizontal and vertical
        flipped_pil = pil_image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
    else:
        raise ValueError(f"Invalid flip_code: {flip_code}. Use 0 (vertical), 1 (horizontal), or -1 (both)")
    
    # Convert back to numpy array
    flipped = np.array(flipped_pil)
    
    return flipped


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image as numpy array
    """
    pil_image = Image.open(str(image_path))
    return np.array(pil_image)


def save_image(image: np.ndarray, output_path: Union[str, Path]) -> None:
    """
    Save an image to file path.
    
    Args:
        image: Image as numpy array
        output_path: Path where to save the image
    """
    if len(image.shape) == 2:
        # Grayscale
        pil_image = Image.fromarray(image, mode='L')
    elif len(image.shape) == 3:
        if image.shape[2] == 3:
            pil_image = Image.fromarray(image, mode='RGB')
        elif image.shape[2] == 4:
            pil_image = Image.fromarray(image, mode='RGBA')
        else:
            # Single channel but 3D array
            pil_image = Image.fromarray(image.squeeze(), mode='L')
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    
    pil_image.save(str(output_path))


def augment_image_pipeline(image_path: Union[str, Path], 
                          output_dir: Union[str, Path] = None,
                          save_original: bool = False) -> list:
    """
    Complete augmentation pipeline for one image:
    1. Load image
    2. Rotate 4 times by 90 degrees (0°, 90°, 180°, 270°)
    3. Flip horizontally
    4. Rotate flipped image 4 times by 90 degrees
    5. Save all variations
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save augmented images (default: same as input)
        save_original: Whether to save the original image in output dir
        
    Returns:
        List of paths to saved augmented images
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Set output directory
    if output_dir is None:
        output_dir = image_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load image
    image = load_image(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Get base filename without extension
    base_name = image_path.stem
    extension = image_path.suffix
    
    saved_paths = []
    
    # Save original if requested
    if save_original:
        original_path = output_dir / f"{base_name}_original{extension}"
        save_image(image, original_path)
        saved_paths.append(original_path)
    
    # Step 1: Rotate original image 4 times by 90 degrees
    angles = [90, 180, 270]
    for i, angle in enumerate(angles):
        rotated = rotate_image(image, angle)
        output_path = output_dir / f"{base_name}_rot{angle:03d}{extension}"
        save_image(rotated, output_path)
        saved_paths.append(output_path)
        logging.debug(f"Saved rotated image ({angle}°): {output_path}")
    
    # Step 2: Flip horizontally
    flipped = flip_image(image, flip_code=1)
    flipped_path = output_dir / f"{base_name}_flipped{extension}"
    save_image(flipped, flipped_path)
    saved_paths.append(flipped_path)
    logging.debug(f"Saved flipped image: {flipped_path}")
    
    # Step 3: Rotate flipped image 4 times by 90 degrees
    for i, angle in enumerate(angles):
        rotated_flipped = rotate_image(flipped, angle)
        output_path = output_dir / f"{base_name}_flipped_rot{angle:03d}{extension}"
        save_image(rotated_flipped, output_path)
        saved_paths.append(output_path)
        logging.debug(f"Saved flipped+rotated image ({angle}°): {output_path}")
    
    logging.debug(f"Augmentation complete! Created {len(saved_paths)} images from {image_path}")
    return saved_paths


def batch_augment_directory(input_dir: Union[str, Path], 
                           output_dir: Union[str, Path] = None,
                           image_extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp'),
                           save_original: bool = False) -> dict:
    """
    Apply augmentation pipeline to all images in a directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save augmented images (default: same as input)
        image_extensions: Tuple of valid image file extensions
        save_original: Whether to save original images in output dir
        
    Returns:
        Dictionary mapping original image paths to lists of augmented image paths
    """
    input_dir = Path(input_dir)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Find all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    if not image_files:
        logging.debug(f"No image files found in {input_dir} with extensions {image_extensions}")
        return results
    
    logging.debug(f"Found {len(image_files)} images to process...")
    
    import concurrent.futures, tqdm

    def process_image(image_path):
        try:
            logging.debug(f"\nProcessing: {image_path}")
            augmented_paths = augment_image_pipeline(
                image_path,
                output_dir,
                save_original=save_original
            )
            return str(image_path), [str(p) for p in augmented_paths]
        except Exception as e:
            logging.debug(f"Error processing {image_path}: {e}")
            return str(image_path), []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_image, img_path): img_path for img_path in image_files}
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures)):
            img_path, augmented = future.result()
            results[img_path] = augmented
    
    logging.debug(f"\nBatch augmentation complete! Processed {len(results)} images.")
    return results


# Additional utility functions
def rotate_90_multiple(image: Union[np.ndarray, Image.Image], times: int = 1) -> np.ndarray:
    """
    Rotate image by 90 degrees multiple times.
    
    Args:
        image: Input image
        times: Number of 90-degree rotations (1-3)
        
    Returns:
        Rotated image as numpy array
    """
    times = times % 4  # Ensure 0-3 range
    if times == 0:
        return image.copy() if isinstance(image, np.ndarray) else np.array(image)
    else:
        # Use rotate_image with appropriate angle
        angle = times * 90
        return rotate_image(image, angle)


def flip_vertical(image: Union[np.ndarray, Image.Image]) -> np.ndarray:
    """Flip image vertically."""
    return flip_image(image, flip_code=0)


def flip_horizontal(image: Union[np.ndarray, Image.Image]) -> np.ndarray:
    """Flip image horizontally."""
    return flip_image(image, flip_code=1)


def flip_both(image: Union[np.ndarray, Image.Image]) -> np.ndarray:
    """Flip image both horizontally and vertically."""
    return flip_image(image, flip_code=-1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Image augmentation for terrain prediction dataset"
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to input image file or directory"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        nargs="?",
        default=None,
        help="Directory to save augmented images (default: same as input)"
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Keep original image(s) in output directory"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="KClean augmented images"
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_dir = args.output_dir
    save_original = args.keep_original
    clean_augmented = args.clean
    
    if clean_augmented:
        for pattern in ['*_original*', '*_rot*', '*_flipped*']:
            for file in Path(output_dir).glob(pattern):
                try:
                    file.unlink()
                    logging.debug(f"Deleted file: {file}")
                except Exception as e:
                    logging.error(f"Error deleting file {file}: {e}")


    if input_path.is_file():
        logging.debug(f"Processing single image: {input_path}")
        augment_image_pipeline(input_path, output_dir, save_original=save_original)
    elif input_path.is_dir():
        logging.debug(f"Processing directory: {input_path}")
        batch_augment_directory(input_path, output_dir, save_original=save_original)
    else:
        logging.debug(f"Invalid path: {input_path}")
        parser.logging.info_help()
