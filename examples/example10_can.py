"""

Sensitivity is given by equation (12) as

                    1 - C(s) * G_m(s) * exp(-s*tau_m) 
    S(s) = -------------------------------------------------------------
            1 + C(s) * ( G_i(s) * exp(-s*tau) - G_m(s) * exp(-s*tau_m))

with:
            1
    D(s) = --- (a_0 + a_1 * exp(-s * theta) + ... a_N * exp(-s * N * theta))
            s

and:

               K
    G(s) = --------- * exp(-s*tau_m)
            T*s + 1

"""

import numpy as np

region1 = (-10, 1, 0.1, 5200)

# Gi
Gi_num = np.array([[1.816*1e6, 1.055*1e5, 2286.]])
Gi_denum = np.array([[3.079*1e6, 2.581*1e5, 9647, 167.5, 1.]])

#
K = 0.59
T = 0.018

# D(s)
theta = 0.01
N = 60
tau_vector = theta * np.arange(0, N, 1, dtype=np.float64)
gains = np.array([4.0445916828132935, -8.229999128255557, 4.499515802677762, 21.391081689487272,
                  -3.609205673017209, -21.18413780071207, 0.3194135628530959, 6.449115172602704,
                  -5.437067253522206, 0.7794203418966514, 3.815245585637833, -4.108654119266066,
                  0.4999079596910827, 2.9408420077422943, -3.4349766762750207, 0.6175361714500549,
                  2.541792176172895, -3.368184912650808, 0.856802524947441, 2.5102717837043107,
                  -3.788517624552782, 1.2296915996576072, 2.9687942285164017, -5.003307087891964,
                  1.512758577502754, 4.119076358830111, -8.155514452238766, 4.574000478694591,
                  21.465566365504078, -3.534720997000289, -21.109653124695264, 0.39389823886987546,
                  6.5235998486195035, -5.362582577505395, 0.8539050179135029, 3.889730261654673, 
                  -4.03416944324924, 0.5743926357079403, 3.015326683759137, -3.3604920002582004,
                  0.6920208474668526, 2.6162768521897273, -3.2937002366339647, 0.9312872009642118,
                  2.5847564597211043, -3.7140329485359427, 1.3041762756744197, 3.0432789045332393,
                  -4.928822411875137, 1.5872432535195538, 4.193561034846927, -8.081029776221925,
                  4.64848515471137, 21.54005104152088, -3.460236320983662, -21.035168448678412,
                  0.46838291488678796, 6.598084524636295, -5.288097901488551, 0.9283896939302438])



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import qpmr
    import qpmr.plot

    logger = qpmr.init_logger(level="DEBUG")


    Gi = qpmr.TransferFunction(
        num=qpmr.QuasiPolynomial.from_array(Gi_num),
        denum=qpmr.QuasiPolynomial.from_array(Gi_denum),
    )
    D = qpmr.QuasiPolynomial(gains[:, np.newaxis], tau_vector)
    C = qpmr.TransferFunction(
        num=qpmr.QuasiPolynomial.from_array(np.array([1, T])),
        denum=qpmr.QuasiPolynomial.from_array(np.array([0, K])),
    ) * qpmr.QuasiPolynomial(gains[:, np.newaxis], tau_vector)

    Gm = qpmr.TransferFunction(
        num=qpmr.QuasiPolynomial.from_array(np.array([[K]])),
        denum=qpmr.QuasiPolynomial.from_array(np.array([T, 1])),
    )

    

    # print(C.num.coefs, C.num.delays)
    # print(C.denum.coefs, C.denum.delays)
    # print(S.coefs, S.delays)




    # coefs
    # vec = np.full_like(a_vector, fill_value=0.0)
    # vec[0] = gamma

    # coefs = np.c_[a_vector, vec]
    # delays = tau_vector

    roots, meta = qpmr.qpmr(region1, D.coefs, D.delays)
    complex_grid = meta.complex_grid
    value = meta.z_value
    
    fig, ax = plt.subplots(1,1,figsize=(12,6))
    qpmr.plot.roots_basic(roots, ax=ax)
    plt.show()







