# postprocess.py
# Postprocessing utilities

import os
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.transform import resize
from skimage.io import imsave
from scipy.interpolate import griddata
from tqdm import tqdm

def postprocess(output, config, save_dir=None, show=True):
    """
    Postprocesses model output tensor using wrap_img and plots/saves results.
    Args:
        output: torch.Tensor or np.ndarray, shape (N, 1, H, W) or (N, H, W)
        config: configuration object or dict
        save_dir: directory to save postprocessed images (optional)
        show: whether to display images interactively
    Returns:
        wrapped: np.ndarray, postprocessed images
    """
    import numpy as np
    if hasattr(output, 'detach'):
        pred_np = output.detach().cpu().numpy().squeeze(1)
    else:
        pred_np = output
    out_dir = save_dir or config.get('wrap_pred_dir', './wrap_pred')
    os.makedirs(out_dir, exist_ok=True)
    wrapped = wrap_img(
        pred_np,
        out_dir,
        (256, 256),
        config.get('base_name', 'model_pred'),
        indikator=True
    )
    
    return wrapped


def wrap_img(images: list, out_dir: str, img_size:tuple , base_name:str, indikator:bool):
    """
    Funkcija koja unwrap slike vraca u originalni radijalni oblik
    out_dir: direktorijum u kojem cemo cuvati kreirane slike
    tensor_unwrap: tensor sa slikama za obradu
    img_size: veličina slike (tuple)
    base_name: osnova za string u nazivu slike
    indikator: da li je slika u opsegu 0-1 (True) ili 0-255 (False)
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    wrap_img_tensor = []
    for i in tqdm(range(images.shape[0]), desc='Wrapping images'):
        unwrap_img = np.squeeze(images[i, :, :])
        center = (unwrap_img.shape[0] / 2, unwrap_img.shape[1] / 2)
        wrp_img = unwrap_img#radial_wrap(unwrap_img, img_size, center)
        
        wrp_img = wrp_img * 255 if indikator else wrp_img
        wrp_img = wrp_img.astype(np.uint8)
        output_image_name = os.path.join(out_dir, f"{base_name}_wrap{i+1:03d}.png")

        img_res = resize(
            wrp_img, (256, 256), order=1, preserve_range=True
        ).astype(np.uint8)
        imsave(output_image_name, img_res)
        
    return wrap_img_tensor


def radial_wrap(transformed_img, img_size, center):
    """
    Rekonstruiše sliku iz unwrap-ovanog oblika uz interpolaciju unutar kruga
    transformed_img: matrica: kolone = pravci, redovi = rastojanje od centra
    img_size: veličina originalne slike [visina, širina]
    center: koordinata centra [yc, xc]
    """
    num_radii, num_angles = transformed_img.shape
    theta = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    max_r = min(
        center[0], center[1], img_size[0] - center[0], img_size[1] - center[1]
    )
    r = np.linspace(0, max_r, num_radii)
    img = np.zeros(img_size)
    weight = np.zeros(img_size)
    for i in range(num_angles):
        for j in range(num_radii):
            x = center[1] + r[j] * np.cos(theta[i])
            y = center[0] + r[j] * np.sin(theta[i])
            xi = int(round(x))
            yi = int(round(y))
            if 0 <= xi < img_size[1] and 0 <= yi < img_size[0]:
                img[yi, xi] += transformed_img[j, i]
                weight[yi, xi] += 1
    weight[weight == 0] = np.nan
    img = img / weight
    XX, YY = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
    dist_from_center = np.sqrt((XX - center[1]) ** 2 + (YY - center[0]) ** 2)
    circle_mask = dist_from_center <= max_r
    Y_known, X_known = np.where(~np.isnan(img))
    V_known = img[~np.isnan(img)]
    points = np.stack((X_known, Y_known), axis=-1)
    interp_values = griddata(points, V_known, (XX, YY), method='nearest')
    img[circle_mask & np.isnan(img)] = interp_values[
        circle_mask & np.isnan(img)
    ]
    img[~circle_mask] = 0
    return img
