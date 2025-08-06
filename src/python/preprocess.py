import os
import numpy as np
from skimage.transform import resize
from skimage.io import imread, imsave

from scipy.interpolate import RegularGridInterpolator
from config import get_config
from joblib import Parallel, delayed
from tqdm import tqdm

import argparse
def preprocess_data(config):
    unwrap_img(
        config.get('data.wrap_dir'),
        config.get('data.unwrap_dir'),
        config.get('data.num_angles', 256),
        config.get('data.num_radii', 256),
        (256, 256),
        config.get('data.make_static', 'PL')
    )

def normalize_img(img):
    return (img - np.mean(img)) / np.std(img)
def make_static(image, base):
    return np.clip(normalize_img(image) - normalize_img(base), a_min=0, a_max=255)
    
    
def process_single_image(file_name, input_dir, output_dir, num_angles, num_radii, img_size, statify, base_img_path):
    full_path = os.path.join(input_dir, file_name)
    img = imread(full_path)

    center = (img.shape[0] / 2, img.shape[1] / 2)
    unwrap_img_ = radial_unwrap(img, num_angles, num_radii, center)

    if statify:
        base_img = imread(base_img_path)
        if base_img.shape != img.shape:
            base_img = resize(base_img, img_size, order=1, preserve_range=True).astype(np.uint8)
        img = make_static(img, base_img)

    out_img_name = f"{file_name.split('.')[0]}_unwrap.png"
    fileName = os.path.join(output_dir, out_img_name)
    img_out = unwrap_img_.astype(np.uint8)
    if img_out.shape[0] != 256 or img_out.shape[1] != 256:
        img_out = resize(
            img_out, img_size, order=1, preserve_range=True
        ).astype(np.uint8)
    imsave(fileName, img_out)

def unwrap_img(input_dir, output_dir, num_angles, num_radii, img_size, statify=False, base_img_path="data/base/out_unwrapped/no_obstacle_unwrapped.png", n_jobs=-1):
    """
    Kreira unwrap slike sa zadatim parametrima, radi resize na 256x256 i cuva ih kao
    .png sa pikselima u opsegu 0-255.
    Takodje cuva i tensore (jer proracun dugo traje, pa da mi ostanu za svaki
    slucaj sacuvani).
    Tensor sadrzi podatke dobijene interpolacijom, znaci, oko vrednosti u
    opsegu 0-255, ali realni brojevi, nisu int jer se radi interpolacija.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_list = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])

    Parallel(n_jobs=n_jobs)(
        delayed(process_single_image)(
            file_name, input_dir, output_dir, num_angles, num_radii, img_size, statify, base_img_path
        ) for file_name in tqdm(file_list)
    )


def apply_circular_mask(img):
    """
    Primeni kružnu masku na sliku
    Ulaz:
        img - Ulazna slika (matrica, grayscale)
    Izlaz:
        masked_img - Slika sa NaN pikselima van kruga
    """
    height, width = img.shape
    center = (round(height / 2), round(width / 2))
    max_r = min(center[0], center[1], height - center[0], width - center[1])
    XX, YY = np.meshgrid(np.arange(width), np.arange(height))
    dist_from_center = np.sqrt((XX - center[1]) ** 2 + (YY - center[0]) ** 2)
    circle_mask = dist_from_center <= max_r
    masked_img = img.copy()
    masked_img[~circle_mask] = np.nan
    return masked_img


def radial_unwrap(img, num_angles, num_radii, center):
    """
    Transform image from polar coordinates to rectangular form.
    Optimized for speed by vectorizing coordinate generation and interpolation.
    """
    theta = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    max_r = min(
        center[0], center[1], img.shape[0] - center[0],
        img.shape[1] - center[1]
    )
    r = np.linspace(0, max_r, num_radii)
    img = img.astype(float)
    y = np.arange(img.shape[0])
    x = np.arange(img.shape[1])
    interpolator = RegularGridInterpolator((y, x), img, bounds_error=False, fill_value=0)

    # Vectorized meshgrid for all (r, theta) pairs
    rr, tt = np.meshgrid(r, theta, indexing='ij')
    xq = center[1] + rr * np.cos(tt)
    yq = center[0] + rr * np.sin(tt)
    coords = np.stack([yq, xq], axis=-1).reshape(-1, 2)
    transformed_img = interpolator(coords).reshape(num_radii, num_angles)
    return transformed_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess data for image unwrapping.')
    
    parser.add_argument('--input_dir', type=str, default='./data/PL_wrap_orig',
                        help='Directory containing input images for unwrapping')
    parser.add_argument('--output_dir', type=str, default='./data/PL_preprocessed',
                        help='Directory to save unwrapped images')
    parser.add_argument('--num_angles', type=int, default=256,
                        help='Number of angles for unwrapping')
    parser.add_argument('--num_radii', type=int, default=256,
                        help='Number of radii for unwrapping')
    parser.add_argument('--img_size', type=tuple, default=(256, 256),
                        help='Size of the output images (height, width)')
    parser.add_argument('--statify', action="store_true", 
                        help='Size of the output images (height, width)')
    args= parser.parse_args()
    unwrap_img(args.input_dir, args.output_dir, args.num_angles, args.num_radii, args.img_size, args.statify)
    