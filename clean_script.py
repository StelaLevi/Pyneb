#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 11:24:26 2025

@author: stela
"""

import pyneb as pn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. 📜 Leitura e normalização dos dados do arquivo original
df = pd.read_csv('obs.dat', sep=r'\s+')
print("Colunas detectadas:", df.columns)

# Normalização pela linha Hβ (H1r_4861A)
fluxo_hbeta = df.loc[df['LINE'] == 'H1r_4861A', df.columns[1]].values[0]
df['FLUX_NORMALIZADO'] = df[df.columns[1]] / fluxo_hbeta

# Salvando arquivo com fluxos normalizados
df[['LINE', 'FLUX_NORMALIZADO']].to_csv('obs_norm.dat', sep='\t', index=False)

# 2.Leitura com PyNeb dos fluxos normalizados
obs = pn.Observation()
obs.readData('obs_norm.dat', fileFormat='lines_in_rows', err_default=0.05)

# 3.Correção por extinção interestelar
obs.def_EBV(label1="H1r_6563A", label2="H1r_4861A", r_theo=2.85)
obs.extinction.law = 'F99'   #usar a lei do Fitzpatrick 1999
obs.correctData(normWave=4861)

print(" Linhas carregadas:")
print(obs.getSortedLines())

print(" Fluxos corrigidos:")
obs.printIntens()  # NÃO use print(...) aqui!

# 4.Diagnósticos de temperatura e densidade eletrônica
diags = pn.Diagnostics()
diags.addDiag([
    '[NII] 5755/6584', 
    '[OIII] 4363/5007', 
    '[SII] 6731/6716',
    '[ArIV] 4740/4711',
    '[ClIII] 5538/5518',
    '[ArIV] 7230+/4720+'
])

# Avaliação dos diagramas com observações corrigidas
emisgrids = pn.getEmisGridDict(atom_list=diags.getUniqueAtoms(), den_max=1e6)
diags.plot(emisgrids, obs, i_obs=0)
plt.show()

# Cálculo direto de Te e Ne com duas razões
Te_N2, Ne_N2 = diags.getCrossTemDen('[NII] 5755/6584+', '[SII] 6731/6716', obs=obs)
Te_O3, Ne_O3 = diags.getCrossTemDen('[OIII] 4363/5007+', '[SII] 6731/6716', obs=obs)
#Te_O, Ne_Ar4 = diags.getCrossTemDen('[OIII] 4363/5007+', '[ArIV] 4740/4711', obs=obs)

print(f" Temperatura eletrônica (Te) do [NII]: {Te_N2:.0f} K")
print(f" Densidade eletrônica (Ne) do [SII] via [NII]: {Ne_N2:.1f} cm⁻³")
print(f" Temperatura eletrônica (Te) do [OIII]: {Te_O3:.0f} K")
print(f" Densidade eletrônica (Ne) do [SII] via [OIII]: {Ne_O3:.1f} cm⁻³")

# 5.  Gráfico das emissividades de [O III] em função da temperatura
o3 = pn.Atom('O', 3)
tem = np.arange(100) * 300 + 300  # T de 300 a 30300 K
lineList = [4363, 4959, 5007]     # Linhas principais de [O III]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim([1.e-30, 5e-20])

for line in lineList:
    emiss = o3.getEmissivity(tem, Ne_O3, wave=line)
    plt.semilogy(tem, emiss, label=f"{line} Å")

plt.xlabel('T$_e$ [K]')
plt.ylabel("j(T) [erg cm$^{-3}$ s$^{-1}$]")
plt.title(f'[O III] emissividades @ N$_e$={Ne_O3:.0f} cm⁻³')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()
