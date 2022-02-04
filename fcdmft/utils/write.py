import numpy as np

def write_dos(filename, freqs, ldos, occupancy=None):
    spin = ldos.shape[0]
    if spin == 1:
        with open(filename+'.dat', 'w') as f:
            if occupancy:
                f.write('# n = %0.12g\n'%(occupancy))
            for w,freq in enumerate(freqs):
                f.write('%0.12g %.12g\n'%(freq, ldos[0][w]))
    else:
        with open(filename+'.dat', 'w') as f:
            if occupancy:
                f.write('# n = %0.12g\n'%(occupancy))
            for w,freq in enumerate(freqs):
                f.write('%0.12g %.12g %.12g %.12g\n'%(freq, ldos[0][w], ldos[1][w],
                                                0.5*(ldos[0][w]+ldos[1][w])))

def write_gf_to_dos(filename, freqs, gf):
    ldos = -1./np.pi * np.trace(gf.imag,axis1=1,axis2=2)
    spin = ldos.shape[0]
    if spin == 1:
        with open(filename+'.dat', 'w') as f:
            for w,freq in enumerate(freqs):
                f.write('%0.12g %.12g\n'%(freq, ldos[0][w]))
    else:
        with open(filename+'.dat', 'w') as f:
            for w,freq in enumerate(freqs):
                f.write('%0.12g %.12g %.12g %.12g\n'%(freq, ldos[0][w], ldos[1][w],
                                                0.5*(ldos[0][w]+ldos[1][w])))

def write_sigma(filename, freqs, sigma):
    sig = np.trace(sigma.imag,axis1=1,axis2=2)
    spin = sig.shape[0]
    if spin == 1:
        with open(filename+'.dat', 'w') as f:
            for w,freq in enumerate(freqs):
                f.write('%0.12g %.12g\n'%(freq, sig[0][w]))
    else:
        with open(filename+'.dat', 'w') as f:
            for w,freq in enumerate(freqs):
                f.write('%0.12g %.12g %.12g %.12g\n'%(freq, sig[0][w], sig[1][w],
                                                0.5*(sig[0][w]+sig[1][w])))

def write_sigma_elem(filename, freqs, sigma):
    spin = sigma.shape[0]
    if spin == 1:
        with open(filename+'.dat', 'w') as f:
            for w,freq in enumerate(freqs):
                f.write('%0.12g %0.12g %.12g\n'%(freq, sigma[0][w].real, sigma[0][w].imag))
    else:
        with open(filename+'.dat', 'w') as f:
            for w,freq in enumerate(freqs):
                f.write('%0.12g %0.12g %.12g %.12g %.12g\n'%(freq, sigma[0][w].real, sigma[0][w].imag,
                                                sigma[1][w].real, sigma[1][w].imag))

