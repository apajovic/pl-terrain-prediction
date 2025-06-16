import os
import numpy as np
from skimage.io import imread, imsave
from skimage.measure import profile_line
import argparse

DATA_DIR = "../../data"

def difraction_and_path_loss(cx, cy, c, m, n, x_center, y_center, s, f):
    """
    Odredjivanje slabljenja usled difrakcije i rastojanja
    cx, cy, c: lists of coordinates and heights
    m, n: receiver pixel coordinates
    x_center, y_center: transmitter pixel coordinates
    s: terrain matrix
    f: frequency (Hz)
    """
    lambda_ = 3e8 / f
    Gr = 1.64
    Gt = 1.64
    c = c[1:-1]
    cx = cx[1:-1]
    cy = cy[1:-1]
    d_tx_rx = np.sqrt((n - x_center) ** 2 + (m - y_center) ** 2) * 50
    L = 20 * np.log10(4 * np.pi * d_tx_rx / lambda_) - 10 * np.log10(Gt * Gr)
    if len(c) == 0:
        J = 0
    else:
        obstacle = np.max(c)
        index = np.argmax(c)
        if index == 0:
            return 0,L
        x_obstacle = cx[index]
        y_obstacle = cy[index]
        d_tx_obs = np.sqrt(
            (x_obstacle - x_center) ** 2 + (y_obstacle - y_center) ** 2
        ) * 50
        h_tx = s[x_center, y_center] + 1.5
        h_rx = s[m, n] + 1.5
        h_p = obstacle
        if (h_tx - h_rx) > 0:
            x = (h_tx - h_rx) * (d_tx_rx - d_tx_obs) / d_tx_rx
            h_ov = h_rx + x
        else:
            x = (h_rx - h_tx) * d_tx_obs / d_tx_rx
            h_ov = h_tx + x
        h_n = np.sqrt(lambda_ * d_tx_obs * (d_tx_rx - d_tx_obs) / d_tx_rx)
        if h_ov - h_p <= h_n:
            h = h_p - h_ov
            ni = h * np.sqrt(
                2 / lambda_ * (1 / d_tx_obs + 1 / (d_tx_rx - d_tx_obs))
            )
            if ni >= -0.78:
                J = 6.9 + 20 * np.log10(
                    np.sqrt((ni - 0.1) ** 2 + 1) + ni - 0.1
                )
            else:
                J = 0
        else:
            J = 0
    
    return J, L


def calculate_PL_mat(s):
    """
    Izracunava propagaciono slabljenje za svaki piksel slike s.
    """
    PL_matrix = np.zeros_like(s)
    num_rows, num_cols = s.shape
    f = 700e6  # Hz
    x_center = 128
    y_center = 128
    for m in range(num_rows):
        for n in range(num_cols):
            if n == x_center and m == y_center:
                PL_rx = -(
                    10 * np.log10(1.64) + 10 * np.log10(1.64)
                    + 20 * np.log10((3e8 / f) / (4 * np.pi * 25))
                )
                PL_matrix[m, n] = PL_rx
            else:
                c = profile_line(s, (y_center, x_center), (m, n), mode='constant', cval=0)
                length = int(np.hypot(m - y_center, n - x_center)) + 1
                cy = np.linspace(y_center, m, length)
                cx = np.linspace(x_center, n, length)
                
                J, L = difraction_and_path_loss(cx, cy, c, m, n, x_center, y_center, s, f)
                PL_rx = L + J
                PL_matrix[m, n] = PL_rx
                
                pass
    return PL_matrix


def create_PL_data(input_dir, out_dir):
    """
    Na osnovu generisanih slika sa preprekama generise matrice sa odgovarajucim
    PL proracunima i cuva u folderu.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    file_list = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    for file_name in file_list:
        print(f"Processing file: {file_name}")
        full_path = os.path.join(input_dir, file_name)
        img = imread(full_path).astype(float)
        PL_matrix = calculate_PL_mat(img)
        imsave(os.path.join(out_dir, f"PL_{file_name}"), PL_matrix.astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description="Generate path loss images from terrain images.")
    parser.add_argument('--input_dir', type=str, help='Directory containing input terrain images.')
    parser.add_argument('--out_dir', type=str, help='Directory to save output path loss images.')
    args = parser.parse_args()

    
    create_PL_data(args.input_dir, args.out_dir)
    
if __name__ == "__main__":
    main()