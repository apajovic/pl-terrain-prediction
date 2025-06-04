import os
import numpy as np
from skimage.transform import resize
from skimage.io import imread, imsave
from scipy.interpolate import RegularGridInterpolator


def unwrap_img(dir_unwrap, folder_path, num_angles, num_radii, img_size, base):
    """
    Kreira unwrap slike sa zadatim parametrima, radi resize na 256x256 i cuva ih kao
    .png sa pikselima u opsegu 0-255.
    Takodje cuva i tensore (jer proracun dugo traje, pa da mi ostanu za svaki
    slucaj sacuvani).
    Tensor sadrzi podatke dobijene interpolacijom, znaci, oko vrednosti u
    opsegu 0-255, ali realni brojevi, nisu int jer se radi interpolacija.
    """
    if not os.path.exists(dir_unwrap):
        os.makedirs(dir_unwrap)
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    unwrap_tensor = []
    for file_name in file_list:
        full_path = os.path.join(folder_path, file_name)
        img = imread(full_path)
        center = (img.shape[0] / 2, img.shape[1] / 2)
        unwrap_img_ = radial_unwrap(img, num_angles, num_radii, center)
        unwrap_tensor.append(unwrap_img_)
    unwrap_tensor = np.stack(unwrap_tensor, axis=-1)
    # Cuvanje unwrap slika
    for i in range(unwrap_tensor.shape[-1]):
        img_name = f"{base}_unwrap{i+1:03d}.png"
        fileName = os.path.join(dir_unwrap, img_name)
        img_round = unwrap_tensor[:, :, i].astype(np.uint8)
        if img_round.shape[0] != 256 or img_round.shape[1] != 256:
            img_res = resize(
                img_round, img_size, order=1, preserve_range=True
            ).astype(np.uint8)
            imsave(fileName, img_res)
        else:
            imsave(fileName, img_round)
    # Optionally: save tensor to disk (e.g., np.save)
    return unwrap_tensor


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
    Transformiše sliku iz polarnih pravaca u pravougaoni oblik
    img: ulazna slika (2D matrica)
    num_angles: broj pravaca (ugao rezolucija)
    num_radii: broj rastojanja (koliko uzoraka po pravcu)
    center: (yc, xc) – centar slike
    """
    theta = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    max_r = min(
        center[0], center[1], img.shape[0] - center[0],
        img.shape[1] - center[1]
    )
    r = np.linspace(0, max_r, num_radii)
    transformed_img = np.zeros((num_radii, num_angles))
    img = img.astype(float)
    # Use RegularGridInterpolator instead of interp2d
    y = np.arange(img.shape[0])
    x = np.arange(img.shape[1])
    interpolator = RegularGridInterpolator((y, x), img, bounds_error=False, fill_value=0)
    for i in range(num_angles):
        for j in range(num_radii):
            xq = center[1] + r[j] * np.cos(theta[i])
            yq = center[0] + r[j] * np.sin(theta[i])
            transformed_img[j, i] = interpolator([[yq, xq]])[0]
    return transformed_img
