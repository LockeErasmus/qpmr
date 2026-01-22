""" Internal Model Control scheme based on article [1]

In [1], sensitivity is given by Eq. (12) as

                    1 - C(s) * G_m(s) * exp(-s*tau_m) 
    S(s) = -------------------------------------------------------------
            1 + C(s) * ( G_i(s) * exp(-s*tau) - G_m(s) * exp(-s*tau_m))

we are interested in showing zeros and poles of S(s).
The model is considered in the form of non delayd LTI as

                K
    Gm(s) = ---------
             T*s + 1

Plant Gi(s) was identified as a non delayed LTI of 3rd order (see code bellow)
and controller C(s) is given by

               1
    C(s) = --------- * D(s)
              Gm(s)

with:
            1
    D(s) = --- (a_0 + a_1 * exp(-s * theta) + ... a_N * exp(-s * N * theta))
            s

[1] YUKSEL, Can Kutlu, et al. A distributed delay based controller for
    simultaneous periodic disturbance rejection and input-delay compensation.
    Mechanical Systems and Signal Processing, 2023, 197: 110364.
"""

import numpy as np

# Same regions as in article [1]
region = (-12, 1, -0.1, 5000)

# Gi - identified system
Gi_num = np.array([[1.816*1e6, 1.055*1e5, 2286.]])
Gi_denum = np.array([[3.079*1e6, 2.581*1e5, 9647, 167.5, 1.]])

# Gm - model parameters K and T
K = 0.59
T = 0.018

# delays
tau = 0.2
tau_m = 0.212

# D(s) - input-shaper like structure
theta = 0.01
N = 60
tau_vector = theta * np.arange(0, N, 1, dtype=np.float64)

# these gains were sent original by Can - claimed to be different experiment
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

# these gains were sent as correct ones later
gains = np.array([3.18419888499517, -8.55006730630735, 7.20804319626175, 20.8242855731446,
                  -6.98819718206270, -20.1849552841814, 2.55098297500236, 5.35346406689096,
                  -5.66728253720343, 1.76092216986347, 2.95360281683304, -4.28799405248918,
                  1.39777992979417, 2.34697134257513, -3.57660291251610, 1.38982228186985,
                  1.98505708058481, -3.46991984257449, 1.62166767533749, 1.89645149324260,
                  -3.86994338042140, 2.11291793462551, 2.18097014111035, -5.13335367003861,
                  2.86163128245455, 3.19992555366216, -8.53434063764039, 7.22376986492874,
                  20.8400122418116, -6.97247051339561, -20.1692286155144, 2.56670964366931,
                  5.36919073555793, -5.65155586853645, 1.77664883853048, 2.96932948550004,
                  -4.27226738382219, 1.41350659846120, 2.36269801124215, -3.56087624384912,
                  1.40554895053682, 2.00078374925181, -3.45419317390750, 1.63739434400443,
                  1.91217816190956, -3.85421671175439, 2.12864460329249, 2.19669680977736,
                  -5.11762700137162, 2.87735795112152, 3.21565222232914, -8.51861396897338,
                  7.23949653359569, 20.8557389104785, -6.95674384472881, -20.1535019468474,
                  2.58243631233637, 5.38491740422488, -5.63582919986946, 1.79237550719740])

# expected zeros location for complete vibration suppression
expected_zeros = np.array([4,8,12,16,20,24,28,32]) * 1j * 2 * np.pi

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import qpmr
    import qpmr.plot

    from qpmr.qpmr_v3 import qpmr as qpmr_v3

    logger = qpmr.init_logger(level="DEBUG")

    tau = qpmr.QuasiPolynomial(np.array([[1.]]), np.array([tau]))
    tau_m = qpmr.QuasiPolynomial(np.array([[1.]]), np.array([tau_m]))
    Gi = qpmr.TransferFunction(
        num=qpmr.QuasiPolynomial.from_array(Gi_num),
        denum=qpmr.QuasiPolynomial.from_array(Gi_denum),
    )
    D = qpmr.QuasiPolynomial(gains[:, np.newaxis], tau_vector)
    C = qpmr.TransferFunction(
        num=qpmr.QuasiPolynomial.from_array(np.array([1, T])), # T*s + 1
        denum=qpmr.QuasiPolynomial.from_array(np.array([0, K])), # K*s
    ) * D
    Gm = qpmr.TransferFunction(
        num=qpmr.QuasiPolynomial.from_array(np.array([K])),
        denum=qpmr.QuasiPolynomial.from_array(np.array([1, T])),
    )

    tf = (1 - C * Gm * tau_m) / (1 + C * (Gi * tau - Gm * tau_m) )

    poles, _ = qpmr_v3(region, tf.denum.coefs, tf.denum.delays)    
    zeros, _ = qpmr_v3(region, tf.num.coefs, tf.num.delays)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,10))
    ax1.scatter(expected_zeros.real, expected_zeros.imag, marker="o", s=60, edgecolors="k", facecolors='none')
    ax1 = qpmr.plot.pole_zero(poles, zeros, ax=ax1)
    ax1.set_xlim((-15, 1))
    ax1.set_ylim((0, 5000))
    ax2.scatter(expected_zeros.real, expected_zeros.imag, marker="o", s=60, edgecolors="k", facecolors='none')
    ax2 = qpmr.plot.pole_zero(poles, zeros, ax=ax2)
    ax2.set_xlim((-5, 1))
    ax2.set_ylim((-10, 250)) 
    
    plt.show()