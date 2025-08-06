import os
import numpy as np
from skimage.io import imread, imsave
from skimage.measure import profile_line
import argparse
from skimage.draw import line
from joblib import Parallel, delayed
import time

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
    c = c[1:]
    cx = cx[1:]
    cy = cy[1:]
    d_tx_rx = np.sqrt((n - x_center) ** 2 + (m - y_center) ** 2) * 50
    if len(c) == 0:
        J = 0
    else:
        obstacle = np.max(c)
        index = np.argmax(c)
        if index == 0 or index > (len(cx) - 1) or index > (len(cy) - 1):
            return 0
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
    
    return J


def calculate_PL_mat(img:np.ndarray) -> np.ndarray:
    """
    Izracunava propagaciono slabljenje za svaki piksel slike img.
    """
    PL_matrix = np.zeros_like(img)
    num_rows, num_cols = img.shape
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
                c = profile_line(img, (y_center, x_center), (m, n), mode='constant', cval=0)
                length = int(np.hypot(m - y_center, n - x_center)) + 1
                cy = np.linspace(y_center, m, length)
                cx = np.linspace(x_center, n, length)
                
                J, L = difraction_and_path_loss(cx, cy, c, m, n, x_center, y_center, img, f)
                PL_rx = L + J
                PL_matrix[m, n] = PL_rx
                
                pass
    return PL_matrix


def calculate_PL_mat_parallel(img: np.ndarray, static=False) -> np.ndarray:
    """
    Izracunava propagaciono slabljenje za svaki piksel slike img.
    Optimizovano za performanse: koristi vektorizaciju gde je moguće.
    """
    num_rows, num_cols = img.shape
    f = 700e6  # Hz
    x_center = 128
    y_center = 128

    # Pre-allocate output
    PL_matrix = np.zeros((num_rows, num_cols), dtype=np.float32)

    # Meshgrid for all coordinates
    m_grid, n_grid = np.meshgrid(np.arange(num_rows), np.arange(num_cols), indexing='ij')

    # Calculate distances from center
    dists = np.sqrt((n_grid - x_center) ** 2 + (m_grid - y_center) ** 2) * 50

    # Calculate L for all pixels
    lambda_ = 3e8 / f
    Gr = 1.64
    Gt = 1.64
    if not static:
        with np.errstate(divide='ignore'):
            L = 20 * np.log10(4 * np.pi * dists / lambda_) - 10 * np.log10(Gt * Gr)
        L[dists == 0] = -(
            10 * np.log10(Gt) + 10 * np.log10(Gr)
            + 20 * np.log10((3e8 / f) / (4 * np.pi * 25))
        )
    else:
        L = np.zeros_like(dists, dtype=np.float32)

    # For each pixel, calculate J (diffraction loss)
    # This part is hard to vectorize due to profile_line usage (which is inherently sequential).
    # So, we keep this loop, but only for J.
    # Full vectorization is not feasible here because each pixel requires a unique set of coordinates along the line,
    # and the number of points per line varies. This makes it hard to batch the computation.
    # However, you can parallelize the outer loop using joblib or numba's prange for multi-core CPUs.

    # Example using joblib for parallelism:

    def process_pixel(m, n):
        if n == x_center and m == y_center:
            return L[m, n]
        rr, cc = line(y_center, x_center, m, n)
        mask = (rr >= 0) & (rr < num_rows) & (cc >= 0) & (cc < num_cols)
        rr = rr[mask]
        cc = cc[mask]
        c = img[rr, cc]
        cy = rr
        cx = cc
        J = difraction_and_path_loss(cx, cy, c, m, n, x_center, y_center, img, f)
        return L[m, n] + J

    # Flatten indices for parallel processing
    indices = [(m, n) for m in range(num_rows) for n in range(num_cols)]
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(process_pixel)(m, n) for m, n in indices
    )
    PL_matrix = np.array(results, dtype=np.float32).reshape(num_rows, num_cols)
    return PL_matrix


def create_PL_data(input_dir, out_dir, use_parallel=True, static=False):
    """
    Na osnovu generisanih slika sa preprekama generise matrice sa odgovarajucim
    PL proracunima i cuva u folderu.
    Prikazuje ukupno i prosečno vreme obrade.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    file_list = [f for f in os.listdir(input_dir) if f.endswith('.png')]

    def process_file(file_name):
        print(f"Processing file: {file_name}")
        full_path = os.path.join(input_dir, file_name)
        img = imread(full_path).astype(float)
        PL_matrix = calculate_PL_mat_parallel(img, static=static)
        imsave(os.path.join(out_dir, f"PL_{file_name}"), PL_matrix.astype(np.uint8))

    start_time = time.time()
    Parallel(n_jobs=-1)(
        delayed(process_file)(file_name) for file_name in file_list
    )
    total_time = time.time() - start_time
    avg_time = total_time / len(file_list) if file_list else 0
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per file: {avg_time:.2f} seconds")


def main():
    parser = argparse.ArgumentParser(description="Generate path loss images from terrain images.")
    parser.add_argument('-i', '--input_dir', type=str, help='Directory containing input terrain images.')
    parser.add_argument('-o', '--out_dir', type=str, help='Directory to save output path loss images.')
    parser.add_argument('-s', '--static', action='store_true', help='Subtract vacuum pathloss.')
    args = parser.parse_args()

    
    create_PL_data(args.input_dir, args.out_dir, static=args.static)
    
if __name__ == "__main__":
    main()