#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 16:47:12 2025

@author: stela
"""

#Correção do código

import pandas as pd
import numpy as np
import pyneb as pn
import re

# === 1. Carrega os dados ===
arquivo = '/home/stela/Documents/artigos  i.c/teste_dos_dados.ods'
df = pd.read_excel(arquivo,header=0)
df.set_index(df.columns[0], inplace=True)

# === 2. Transpõe a tabela e corrige nomes dos objetos ===
dc = df.T.copy()
#dc.columns = ["MaC_1-16_1", "MaC_1-16_2", "MaC_1-16_3"]  # Aplica os nomes corretos

# === 3. Cálculo automático de E(B–V) ===
I_theo_HaHb = 2.86  # valor teórico da razão Hα / Hβ

linha_Ha = [i for i in dc.index if "H1r_6563A" in i][0]
linha_Hb = [i for i in dc.index if "H1r_4861A" in i][0]

F_Ha_all = dc.loc['H1r_6563A']
F_Hb_all = dc.loc['H1r_4861A']

# Corrigir a equação pra o valor da lei de extinção para HA
I_obs_HaHb = F_Ha_all / F_Hb_all


# Correct based on the given law and the observed Ha/Hb ratio
RC = pn.RedCorr(law='CCM89')

ratio = np.array(I_obs_HaHb / I_theo_HaHb)



RC.setCorr(ratio, 6563., 4861.)

wave = np.array([3869, 3889, 3968, 3670, 4102, 4341, 4363, 4471, 4686, 4711, 4740, 4861, 4959, 5007, 5200, 5309, 5411, 5518, 5538, 5755, 5876, 6300, 6312, 6364, 6435, 6548, 6563, 6584, 6678, 6716, 6731, 7005, 7065, 7136, 7170, 7237, 7263, 7319, 7751])


I_obs = dc
I_obs = I_obs.to_numpy()   
I_corr = I_obs * RC.getCorrHb(wave)

print(RC.E_BV)


