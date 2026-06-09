from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import _conc_mos
import numpy as np
from pyscf import lib

def get_unit_eri_fast_onthefly(cell, mydf, C_ao_lo):
    kpts = mydf.kpts
    spin, nkpts, nao, nmo = C_ao_lo.shape

    kscaled = cell.get_scaled_kpts(kpts)
    kscaled -= kscaled[0]

    eri = np.zeros((spin * (spin+1) // 2,nmo,nmo,nmo,nmo))
    for kL in range(nkpts):
        if spin == 1:
            Lij = 0j
        elif spin == 2:
            Lij_a = 0j
            Lij_b = 0j

        for i, kpti in enumerate(kpts):
            for j, kptj in enumerate(kpts):
                # Find (ki,kj) that satisfies momentum conservation with kL
                kconserv = -kscaled[i] + kscaled[j] + kscaled[kL]
                is_kconserv = np.linalg.norm(np.round(kconserv) - kconserv) < 1e-12
                if is_kconserv:
                    # obtain AO GDF integrals
                    Lpq = []
                    for LpqR, LpqI, sign in mydf.sr_loop([kpti,kptj], compact=False):
                        Lpq.append(LpqR+LpqI*1j)
                    Lpq = np.vstack(Lpq).reshape(-1,nao,nao)
                    Lpq = Lpq.astype(np.complex128)

                    # AO to MO transformation
                    if spin == 1:
                        Lij_out = None
                        moij, ijslice = _conc_mos(C_ao_lo[0,i], C_ao_lo[0,j])[2:]
                        Lij_out = _ao2mo.r_e2(Lpq, moij, ijslice, tao=[], ao_loc=None, out=Lij_out)
                        Lij += 1./nkpts * Lij_out
                    elif spin == 2:
                        moija, ijslicea = _conc_mos(C_ao_lo[0, i], C_ao_lo[0, j])[2:]
                        moijb, ijsliceb = _conc_mos(C_ao_lo[1, i], C_ao_lo[1, j])[2:]
                        Lij_out_a = None
                        Lij_out_b = None
                        Lij_out_a = _ao2mo.r_e2(Lpq, moija, ijslicea, tao=[], ao_loc=None, out=Lij_out_a)
                        Lij_out_b = _ao2mo.r_e2(Lpq, moijb, ijsliceb, tao=[], ao_loc=None, out=Lij_out_b)
                        Lij_a += 1./nkpts * Lij_out_a
                        Lij_b += 1./nkpts * Lij_out_b

        # assemble ERI
        if spin == 1:
            eri[0] += 1./nkpts * lib.dot(Lij.T.conj(), Lij).reshape(nmo,nmo,nmo,nmo).real
        elif spin == 2:
            eri[0] += 1./nkpts * lib.dot(Lij_a.T.conj(), Lij_a).reshape(nmo,nmo,nmo,nmo).real # aaaa
            eri[1] += 1./nkpts * lib.dot(Lij_a.T.conj(), Lij_b).reshape(nmo,nmo,nmo,nmo).real # aabb
            eri[2] += 1./nkpts * lib.dot(Lij_b.T.conj(), Lij_b).reshape(nmo,nmo,nmo,nmo).real # bbbb

    return eri
