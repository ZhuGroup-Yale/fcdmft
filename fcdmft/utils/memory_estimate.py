def check_memory_gw(gw, is_return=False):
    naux = gw.with_df.get_naoaux()
    mol = gw._scf.mol
    nao = mol.nao_nr()
    mem_est = (nao * nao * naux) * 8 / 1e6
    mem_est_gw = mem_est * 4.5
    if mol.max_memory < mem_est_gw:
        raise ValueError('Estimated max_memory is %s MB'%(mem_est_gw))
    if is_return:
        return mem_est_gw

def check_memory_gwbse(gw, is_return=False):
    naux = gw.with_df.get_naoaux()
    mol = gw._scf.mol
    nao = mol.nao_nr()
    mem_est = (nao * nao * naux) * 8 / 1e6
    mem_est_gw = mem_est * 5.5
    if mol.max_memory < mem_est_gw:
        raise ValueError('Estimated max_memory is %s MB'%(mem_est_gw))
    if is_return:
        return mem_est_gw

def check_memory_rpa(rpa, is_return=False):
    naux = rpa.with_df.get_naoaux()
    mol = rpa._scf.mol
    nao = mol.nao_nr()
    mem_est = (nao * nao * naux) * 8 / 1e6
    mem_est_rpa = mem_est * 2.5
    if mol.max_memory < mem_est_rpa:
        raise ValueError('Estimated max_memory is %s MB'%(mem_est_rpa))
    if is_return:
        return mem_est_rpa

