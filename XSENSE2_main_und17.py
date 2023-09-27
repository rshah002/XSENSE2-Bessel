#==============================================================================
# 
# Bessel Factor Comparison to Eq. 17 
# of Dr. Krafft's Bessel Factor Function Share
#   
#==============================================================================

import matplotlib.pyplot as plt
from math import log,pi
from math import pi,e,cos,sin,sqrt,atan,asin,exp,log
from scipy.special import jv
import numpy as np


mc2 = 511e3                # electron mass [eV]
hbar = 6.582e-16           # Planck's constant [eV*s]
c = 2.998e8                # speed of light [m/s]
lambda0 = 800e-9           # same wavelength for each run
alpha = 1.0/137.0          # fine structure constant
r = 2.818e-15              # electron radius [m]
q = 1.602e-19              # electron charge [C]
eV2J = 1.602176565e-19     # eV --> Joules 
J2eV = 6.242e18            # Joules --> eV
hbarJ = 1.0545718e-34      # Planck's constant [J*s]
me = 9.10938356e-31        # electron mass [kg]
eps0 = 8.85e-12            # epsilon 0 (C^2/Nm^2)


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

if __name__ == '__main__':
    arg_file = open("config.in", "r")
    args = []
    for line in arg_file:
        i = 0
        while (line[i:i + 1] != " "):
            i += 1
        num = float(line[0:i])
        args.append(num)

    #-------------------
    # e-beam parameters
    #-------------------
    En0 = args[0]              # e-beam: mean energy [eV]
    sig_g = args[1]            # e-beam: relative energy spread
    sigma_e_x = args[2]        # e-beam: rms horizontal size [m]
    sigma_e_y = args[3]        # e-beam: rms vertical size [m]
    eps_x_n = args[4]          # e-beam: normalized horizontal emittance [m rad]
    eps_y_n = args[5]          # e-beam: normalized vertical emittance [m rad]

    #------------------
    # Laser parameters
    #------------------
    lambda0 = args[6]          # laser beam: wavelength [m]
    sign = args[7]             # laser beam: normalized sigma [ ]
    sigma_p_x = args[8]        # laser beam: horizontal laser waist size [m]
    sigma_p_y = args[9]        # laser beam: vertical laser waitt size [m]
    a0 = args[10]              # laser beam: field strength a0 []
    iTypeEnv = int(args[11])   # laser beam: laser envelope type
    if (iTypeEnv == 3):        # laser beam: load experimental data -Beth
        data_file = open("Laser_Envelope_Data.txt", "r")
        exp_data = []
        for line in data_file:
           exp_data.append(line.strip().split())
        exp_xi = []
        exp_a = []
        for line in exp_data:
            exp_xi.append(float(line[0]))
            exp_a.append(float(line[1]))
        exp_f=interp1d(exp_xi,exp_a,kind='cubic')       # laser beam: generate beam envelope function
    else:
        exp_xi = []
        exp_f = 0
    modType = int(args[12])    # laser beam: frequency modulation type
    fmd_xi=[]
    fmd_f=[]
    fmdfunc=0
    if (modType == 1):         # exact 1D chirp: TDHK 2014 (f(0)=1)
        a0chirp = a0                # laser beam: a0 chirping value
        fm_param = 0.0
    elif (modType == 2):       # exact 1D chirp: Seipt et al. 2015 (f(+/-inf)=1)
        a0chirp = a0                # laser beam: a0 chirping value
        fm_param = 0.0
    elif (modType == 3):       # RF quadratic chirping
        a0chirp = 0.0
        fm_param = float(args[13])   # laser beam: lambda_RF chirping value
    elif (modType == 4):       # RF sinusoidal chirping
        a0chirp = 0.0
        fm_param = float(args[13])   # laser beam: lambda_RF chirping value
    elif (modType == 5):       # exact 3D chirp: Maroli et al. 2018
        a0chirp = a0                 # laser beam: a0 chirping value
        fm_param = float(args[13])   # p parameter
    elif (modType == 6):       # chirp with ang. dep. (f(0)=1)
        a0chirp = a0                 # laser beam: a0 chirping value
        fm_param = float(args[13])   # theta_FM (optimization angle)
    elif (modType == 7):       # chirp with ang. dep. (f(+/-inf) = 1)
        a0chirp = a0                 # laser beam: a0 chirping value
        fm_param = float(args[13])   # theta_FM (optimization angle)
    elif (modType == 8):       # saw-tooth chirp
        a0chirp = a0                   # laser beam: a0 chirping value
        fm_param = float(args[13])     # chirping slope
    elif (modType == 9):       # read chirping data from a file and generate function -Beth
        data_file = open("Fmod_Data.txt", "r")
        fmod_data = []
        for line in data_file:
           fmod_data.append(line.strip().split())
        fmd_xi = []
        fmd_f = []
        for line in fmod_data:
            fmd_xi.append(float(line[0]))
            fmd_f.append(float(line[1]))
        fmdfunc=interp1d(fmd_xi,fmd_f,kind='cubic')
        a0chirp =  float(args[13])
        lambda_RF = 0.0
    elif (modType == 10):       # chirping from GSU 2013
        a0chirp = a0           
        fm_param = float(args[13])
    else:                      # no chirping
        a0chirp = 0.0
        fm_param = 0.0
    l_angle = args[14]         # laser beam: angle between laser & z-axis [rad]

    #---------------------
    # Aperture parameters
    #---------------------
    TypeAp = args[15]          # aperture: type: =0 circular; =1 rectangular
    L_aper = args[16]          # aperture: distance from IP to aperture [m]
    if (TypeAp == 0):
        R_aper = args[17]      # aperture: physical radius of the aperture [m]
        tmp = args[18]
        theta_max = atan(R_aper/L_aper)
    else:
        x_aper = args[17]
        y_aper = args[18]
 
    #-----------------------
    # Simulation parameters
    #-----------------------
    wtilde_min = args[19]      # simulation: start spectrum [norm. w/w0 units]
    wtilde_max = args[20]      # simulation: end spectrum [norm. w/w0 units]
    Nout = int(args[21])       # simulation: number of points in the spectrum
    Ntot = int(args[22])       # simulation: resolution of the inner intergral
    Npart = int(args[23])      # simulation: number of electron simulated
    N_MC = int(args[24])       # simulation: number of MC samples
    iFile = int(args[25])      # simulation: =1: read ICs from file; <> 1: MC
    iCompton = int(args[26])   # simulation: =1: Compton; <>1: Thomson
    RRmode = int(args[27])     # Radiation reaction model

    #--------------------------------------------
    # Compute basic parameters for the two beams
    #--------------------------------------------
    gamma = En0/mc2
    beta = sqrt(1.0-1.0/(gamma**2))
    c1 = gamma*(1.0 + beta)
    omega0 = 2.0*pi*c1*c1*(c/lambda0)
    ntothm1 = Ntot/2.0 - 1.0
    d_omega = (wtilde_max - wtilde_min)/(Nout-1)

    eps_x = eps_x_n/gamma
    eps_y = eps_y_n/gamma
    sigma_e_px = eps_x/sigma_e_x
    sigma_e_py = eps_y/sigma_e_y
    pmag0 = (eV2J/c)*sqrt(En0**2-mc2**2)
    sigmaPmag = (eV2J/c)*sqrt(((En0*(1.0+sig_g))**2)-mc2**2)-pmag0
    E_laser = 2*pi*hbar*c/lambda0

    omegas_E = []
    Es_E = []
    dNdEs_E = []
    omega_Peaks_E = []
    bessel_Peaks_E = []
    bessel_Peaks_Scaled_E = []
            
    data = open("output_SENSE.txt").readlines()
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
    dNdE_i=np.multiply(dNdE_i,(R_aper**2)/(theta_max**2)) 

    # omegaPeaks_i, peakValues_i = dNdE_peaks(omega_i, dNdE_i)
    besselPeaks_i, besselOmega_i = bessel_factors(omega_i, a0, sign, R_aper, E_laser)
    omega_Peaks_E.append(besselOmega_i)
    bessel_Peaks_E.append(besselPeaks_i)

    print("E =%s eV" %En0, "MeV \t K =", a0)
    print("first SENSE peak / first Bessel peak =", max(dNdE_i) / besselPeaks_i[0])

    plt.title(r'E = %s eV,  $K$ = %s' % (En0, a0))
    plt.plot(omega_i, dNdE_i, '-',besselOmega_i,besselPeaks_i,'o')  # ,besselOmega_i,besselPeaks_i,'o'
    plt.xlabel(r'$\omega / \omega_0$')
    plt.ylabel("dN/dE\'")

    plt.show()
