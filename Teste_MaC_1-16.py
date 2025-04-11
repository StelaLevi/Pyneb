#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 13:17:49 2025

@author: stela
"""

import pandas as pd
import numpy as np
import pyneb as pn
import re

# === 1. Carrega os dados ===
arquivo = '/home/stela/Documents/artigos  i.c/teste_dos_dados.ods'
df = pd.read_excel(arquivo)
df.set_index(df.columns[0], inplace=True)

# === 2. Transpõe a tabela e corrige nomes dos objetos ===
dc = df.T.copy()
dc.columns = ["MaC_1-16_1", "MaC_1-16_2", "MaC_1-16_3"]  # Aplica os nomes corretos

# === 3. Cálculo automático de E(B–V) ===
#f_Ha = 2.535
#f_Hb = 3.609   
R_teo = 2.86  # valor teórico da razão Hα / Hβ

#linha_Ha = [i for i in dc.index if "H1r_6563A" in i][0]
#linha_Hb = [i for i in dc.index if "H1r_4861A" in i][0]

F_Ha_all = dc.loc['H1r_6563A']
F_Hb_all = dc.loc['H1r_4861A']

# Corrigir a equação pra o valor da lei de extinção para HA
R_obs_all = F_Ha_all / F_Hb_all
E_BV_all = (2.5 / (f_Hb - f_Ha)) * np.log10(R_obs_all / R_teo)

# Adiciona E(B–V)
df["E(B-V)"] = E_BV_all.values
dc.loc["E(B-V)"] = E_BV_all
df.loc["E(B-V)"] = E_BV_all

ebv_values = pd.Series(E_BV_all.values, index=dc.columns) # Pega E(B–V) por objeto
# === 4. Prepara os dados para correção ===
fluxos_obs = df.drop(index="E(B-V)")  # Remove a linha E(B-V)
if not all(col in ebv_values.index for col in fluxos_obs.columns):
    fluxos_obs = fluxos_obs.T
redcorr = pn.RedCorr(R_V=3.1, law='CCM89')

# === 5. Extrai os comprimentos de onda das linhas ===
def extrair_lambda(nome_linha):
    match = re.search(r'λ?(\d{4}(?:\.\d+)?)', nome_linha)
    return float(match.group(1)) if match else np.nan

comprimentos_onda = fluxos_obs.index.to_series().apply(extrair_lambda)
linhas_validas = ~comprimentos_onda.isna()

# Filtra apenas linhas com λ válido
comprimentos_onda = comprimentos_onda[linhas_validas]
fluxos_obs = fluxos_obs.loc[linhas_validas]

# === 6. Calcula A(λ)/E(B–V) só uma vez ===
A_lambda_por_unidade = redcorr.getCorr(comprimentos_onda.values)

# === 7. Corrige os fluxos ===
fluxos_corrigidos = fluxos_obs.copy()

for obj in fluxos_obs.columns:
    ebv = ebv_values[obj]
    A_lambda = A_lambda_por_unidade * ebv
    fator_correcao = 10 ** (0.4 * A_lambda)
    fluxos_corrigidos[obj] = fluxos_obs[obj].values * fator_correcao

# === 8. Atualiza as tabelas finais ===
df_corrigido = fluxos_corrigidos.copy()
df_corrigido.loc["E(B-V)"] = ebv_values
dc_corrigido = df_corrigido.T.copy()

# === 9. Temperatura Eletrônica ===

def encontrar_linha(index, alvo):
    """
    Procura uma linha no índice do DataFrame que contenha o comprimento de onda 'alvo'.
    """
    for linha in index:
        linha_limpa = linha.replace(" ", "").replace("λ", "")
        if re.search(str(alvo), linha_limpa):
            return linha
    raise ValueError(f"Linha com λ={alvo} não encontrada!")

    
 # Busca os nomes corretos das linhas
linha_4363 = encontrar_linha(dc_corrigido.columns, 4363)
linha_4959 = encontrar_linha(dc_corrigido.columns, 4959)
linha_5007 = encontrar_linha(dc_corrigido.columns, 5007)

dc_corrigido = dc_corrigido.T

I_4363 = dc_corrigido.loc[linha_4363]
I_4959 = dc_corrigido.loc[linha_4959]
I_5007 = dc_corrigido.loc[linha_5007]
R_O3 = (I_4959 + I_5007) / I_4363
O3 = pn.Atom('O', 3)
densidade = 100  # em cm⁻3
Te_O3 = R_O3.apply(lambda r: O3.getTemDen(int_ratio=r, den=densidade, wave1=5007, wave2=4363))

dc_corrigido.loc["R_O3"] = R_O3
dc_corrigido.loc["Te_O3"] = Te_O3

# === 10. Densidade Eletrônica

linha_6716 = encontrar_linha(dc_corrigido.index, 6716)
linha_6731 = encontrar_linha(dc_corrigido.index, 6731)

I_6716 = dc_corrigido.loc[linha_6716]
I_6731 = dc_corrigido.loc[linha_6731]

R_SII = I_6716 / I_6731

S2 = pn.Atom('S', 2)

ne = S2.getTemDen(int_ratio=R_SII, tem=Te_O3, wave1=6716, wave2=6731)

dc_corrigido.loc["R_SII"] = R_SII
dc_corrigido.loc["ne(SII)"] = ne

# === 11. Abundâncias ionicas ===

def get_parametro(param, objeto):
    """
    Retorna o valor de um parâmetro físico (ex: 'Te_O3') para um objeto específico.
    
    Parâmetros:
    - param: str, o nome do parâmetro (ex: 'Te_O3', 'ne(SII)', etc.)
    - objeto: str, o nome do objeto (ex: 'MaC_1-16_1', 'MaC_1-16_2', etc.)

    Retorna:
    - float: valor do parâmetro solicitado.
    """
    try:
        return dc_corrigido.loc[param, objeto]
    except KeyError:
        print(f"Parâmetro '{param}' ou objeto '{objeto}' não encontrado.")
        return None

print(get_parametro("Te_O3", "MaC_1-16_1"))
print(get_parametro("ne(SII)", "MaC_1-16_3"))

def identificar_ion(linha):
    """
    Exemplo: '[OIII]λ5007' → ('O', 3, 5007.0)
    """
    match = re.match(r'\[?([A-Za-z]+)(?:[IVX]+)?\]?λ(\d+(?:\.\d+)?)', linha.replace(" ", ""))
    if not match:
        return None
    elemento = match.group(1)
    ionizacao = linha.count('I')  # Ex: [OIII] → 3
    onda = float(match.group(2))
    return (elemento, ionizacao, onda)


abundancias = pd.DataFrame(index=dc_corrigido.columns, columns=dc_corrigido.columns)

for linha in dc_corrigido.index:
    info = identificar_ion(linha)
    if not info:
        continue
    elemento, ionizacao, onda = info

    try:
        ion = pn.Atom(elemento, ionizacao)
    except:
        continue  # Se o PyNeb não reconhecer, pula

    for obj in dc_corrigido.columns:
        try:
            intensidade = dc_corrigido.loc[linha, obj]
            tem = get_parametro("Te_O3", obj)
            den = get_parametro("ne(SII)", obj)

            abund = ion.getIonAbundance(intensity=intensidade, tem=tem, den=den, wave=onda)
            abundancias.loc[linha, obj] = abund
        except:
            abundancias.loc[linha, obj] = np.nan

abundancias = abundancias.astype(float)
abundancias.head()  # mostra as primeiras
abundancias.to_csv("abundancias_ionicas.csv")  # salva









