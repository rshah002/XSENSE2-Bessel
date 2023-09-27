#==============================================================================
# 
# Bessel Factor Comparison to Eq. 17 
# of Dr. Krafft's Bessel Factor Function Share
#   
#==============================================================================

import matplotlib.pyplot as plt
from math import log,pi
from scipy.special import jv
import numpy as np


mc2 = 511e3                # electron mass [eV]
hbar = 6.582e-16           # Planck's constant [eV*s]
c = 2.998e8                # speed of light [m/s]
lambda0 = 800e-9           # same wavelength for each run
alpha = 1.0/137.0          # fine structure constant


def bessel_factors(omega, a0, N, R, E_laser):
    # Must Multiply Bessel Factors by pi*R_aper^2 to convert
    # from d^2N/dE'dOmega to dN/dE'
    scale = (alpha * (N ** 2) * pi * (R ** 2)) / (2 * E_laser)
    harmonics = 1 / (1 + ((a0 ** 2) / 2))
    harmonic = harmonics
    besselFacs = []
    besselOmega = []
    i = 0
    while harmonic <= omega[-1]:
        n = (2 * i) + 1
        n1 = (n - 1) / 2
        n2 = (n + 1) / 2
        d = (n * (a0 ** 2)) / (4 * (1 + ((a0 ** 2) / 2)))
        Jn2 = (jv(n1, d) - jv(n2, d)) ** 2
        dNdEB = scale * d * Jn2
        i += 1
        besselFacs.append(dNdEB)
        besselOmega.append(harmonic)
        harmonic = (n+2) * harmonics
    return besselFacs, besselOmega


sig = 10
R = 0.0000001
theta_max = 1.66667e-9 # theta_max
E0s = [530]
a0s = [4, 6, 8 ,20]
gamma = 530.0e6/mc2
E_laser=0.0000123984193

if __name__ == '__main__':
    fig = 1
    Omegas = []
    Es = []
    DNdEs = []
    Omega_Peaks = []
    Bessel_Peaks = []
    Bessel_Peaks_Scaled = []
    for E0 in E0s:
        omegas_E = []
        Es_E = []
        dNdEs_E = []
        omega_Peaks_E = []
        bessel_Peaks_E = []
        bessel_Peaks_Scaled_E = []
        for a0 in a0s:
            data = open("bessel_530_a0_comp_und/a0=%s/output_SENSE.txt" % (a0)).readlines()
            data2 = []
            omega_i = []
            E_i = []
            dNdE_i = []
            dNdE_r_i = []
            for line in data:
                data2.append(line.strip().split())
            data = data2
            for line in data:
                if float(line[0]) != 0:
                    E_i.append(float(line[0]))
                    omega_i.append(float(line[1]))
                    dNdE_i.append(float(line[2]))
            omegas_E.append(omega_i)
            dNdEs_E.append(dNdE_i)

    # Ryan -- Divide by theta_max^2 to cancel out theta_max^2 in
    # yfac scaling in XSENSE.py and multiply by R_aper^2 to cancel
    # out meters^2 and to return dN/dE' in [1/eV]
            dNdE_i=np.multiply(dNdE_i,(R**2)/(theta_max**2)) 

            # omegaPeaks_i, peakValues_i = dNdE_peaks(omega_i, dNdE_i)
            besselPeaks_i, besselOmega_i = bessel_factors(omega_i, a0, sig, R, E_laser)
            omega_Peaks_E.append(besselOmega_i)
            bessel_Peaks_E.append(besselPeaks_i)

            manual_scale = max(dNdE_i) / besselPeaks_i[0]
            bfscale_i = []
            for i in besselPeaks_i:
                bfscale_i.append(manual_scale * i)

            bessel_Peaks_Scaled_E.append(bfscale_i)

            print("E =%s MeV" %E0, "MeV \t K =", a0)
            print("first SENSE peak / first Bessel peak =", max(dNdE_i) / besselPeaks_i[0])

            plt.figure(fig)
            plt.title(r'E = %s MeV,  $K$ = %s' % (E0, a0))
            plt.plot(omega_i, dNdE_i, '-',besselOmega_i,besselPeaks_i,'o')  # ,besselOmega_i,besselPeaks_i,'o'
            plt.xlabel(r'$\omega / \omega_0$')
            plt.ylabel("dN/dE\'")
            fig += 1


        Omegas.append(omegas_E)
        Es.append(Es_E)
        DNdEs.append(dNdEs_E)
        Omega_Peaks.append(omega_Peaks_E)
        Bessel_Peaks.append(bessel_Peaks_E)
        Bessel_Peaks_Scaled.append(bessel_Peaks_Scaled_E)
    plt.show()
