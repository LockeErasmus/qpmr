import numpy as np

N = 10

region = [-4.5, 2.5, 0, 20]
delays = np.array([24.99, 23.35, 19.9, 18.52, 13.32, 10.33, 8.52, 4.61, 0.0])
coefs = np.array([[51.7, 0, 0, 0, 0, 0, 0, 0 , 0],
                    [1.5, -0.1, 0.04, 0.03, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0.5, 0, 0, 0, 0, 0],
                    [0, 25.2, 0, -0.9, 0.2, 0.15, 0, 0, 0],
                    [7.2, -1.4, 0, 0, 0.1, 0, 0.8, 0, 0],
                    [0, 19.3, 2.1, 0, -8.7, 0, 0, 0, 0],
                    [0, 6.7, 0, 0, 0, -1.1, 0, 1, 0],
                    [29.1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, -1.8, 0.001, 0, 0, -12.8, 0, 1.7, 0.2]])
powers = np.arange(0, coefs.shape[1], 1)

if __name__ == "__main__":
    import time
    from qpmr import qpmr
    from qpmr.qpmr_v2_fractional import qpmr_fractional
    from qpmr.utils import init_qpmr_logger

    init_qpmr_logger("DEBUG")

    s1 = time.time()
    for _ in range(N):
        roots, meta = qpmr(region, coefs, delays)
    e1 = time.time()
    
    s2 = time.time()
    for _ in range(N):
        roots, meta = qpmr_fractional(region, coefs, delays, powers)
    e2 = time.time()


    print(f"1: total time: {e1-s1}, per run {(e1-s1)/N}")
    print(f"2: total time: {e2-s2}, per run {(e2-s2)/N}")
