import sys

def rmse_to_dB(pixel):
    pl_min = -162
    pl_max = -75
    return pixel * (pl_max - pl_min)

if __name__ == "__main__":
    test_pixel = float(sys.argv[1])
    print(rmse_to_dB(test_pixel))  # Expected output: [-162.   -118.5 -75. ]