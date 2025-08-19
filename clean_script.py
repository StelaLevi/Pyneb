#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 00:31:28 2025
@author: stela
"""

import pandas as pd
import numpy as np
import pyneb as pn
import os

# ==============================================================================
# --- ETAPA 1: Leitura e normalização dos fluxos pelo H-beta ---
# ==============================================================================
print("--- Etapa 1: Lendo e normalizando os fluxos pelo H-beta ---")
try:
    df_bruto = pd.read_excel('dados_A.ods', engine='odf')
except FileNotFoundError:
    print("-> ERRO CRÍTICO: Arquivo 'dados_A.ods' não encontrado. Saindo.")
    exit()

objetos = df_bruto.columns[1:]
output_dir = 'dados_normalizados'
os.makedirs(output_dir, exist_ok=True)
arquivos_dat = {}

for obj in objetos:
    try:
        linha_hbeta = df_bruto['LINE'] == 'H1r_4861A'
        if not linha_hbeta.any(): raise ValueError("Linha H1r_4861A não encontrada.")
        fluxo_hbeta = df_bruto.loc[linha_hbeta, obj].values[0]
        if pd.isna(fluxo_hbeta) or fluxo_hbeta <= 0: raise ValueError(f"Fluxo de H-beta inválido.")
        
        df_norm = df_bruto[['LINE', obj]].copy()
        df_norm.columns = ['LINE', 'FLUX']
        df_norm['FLUX'] = pd.to_numeric(df_norm['FLUX'], errors='coerce') / fluxo_hbeta
        df_norm.dropna(subset=['FLUX'], inplace=True)
        df_norm = df_norm[df_norm['FLUX'] > 0]
        
        caminho_dat = os.path.join(output_dir, f'{obj}.dat')
        df_norm.to_csv(caminho_dat, sep='\t', index=False, header=False)
        arquivos_dat[obj] = caminho_dat
    except Exception as e:
        print(f"-> Aviso em '{obj}': {e}")

# ==============================================================================
# --- ETAPA 2: Correção pelo avermelhamento ---
# ==============================================================================
print("\n--- Etapa 2: Corrigindo fluxos pelo avermelhamento ---")
obs_corrigidas = {}
for obj, caminho in arquivos_dat.items():
    try:
        obs = pn.Observation(caminho, fileFormat='lines_in_rows', err_default=0.05)
        obs.def_EBV(label1="H1r_6563A", label2="H1r_4861A", r_theo=2.85)
        obs.extinction.law = 'F99'
        obs.correctData(normWave=4861)
        obs_corrigidas[obj] = obs
    except Exception as e:
        print(f"-> Aviso ao corrigir '{obj}': {e}")

# ==============================================================================
# --- ETAPA 3: Diagnóstico Te/Ne robusto ---
# ==============================================================================
print("\n--- Etapa 3: Calculando Te e Ne com lógica de fallback ---")

def obter_diagnostico_robusto(obs, obj_name):
    diags = pn.Diagnostics()
    diags.addDiag(['[NII] 5755/6584', '[OIII] 4363/5007', '[SII] 6731/6716', '[ArIV] 4740/4711', '[ClIII] 5538/5518'])
    Te_nii, Ne_sii, Te_oiii, Ne_cl, Ne_ar = (np.nan,) * 5
    
    try: Te_nii, Ne_sii = diags.getCrossTemDen('[NII] 5755/6584', '[SII] 6731/6716', obs=obs)
    except Exception: pass
    
    try: Te_oiii, Ne_ar = diags.getCrossTemDen('[OIII] 4363/5007', '[ArIV] 4740/4711', obs=obs)
    except Exception: pass
    
    try:
        if np.isnan(Ne_ar):
            _, Ne_cl = diags.getCrossTemDen('[OIII] 4363/5007', '[ClIII] 5538/5518', obs=obs)
    except Exception: pass
    
    ne_final_baixa = Ne_sii
    ne_final_media = Ne_ar if pd.notna(Ne_ar) else Ne_cl
    
    if np.isnan(ne_final_baixa):
        ne_final_baixa = 1000.0
        print(f"-> Fallback em '{obj_name}': Ne_baixa assumida como {ne_final_baixa:.0f} cm⁻³")
    if np.isnan(ne_final_media):
        ne_final_media = ne_final_baixa
        print(f"-> Fallback em '{obj_name}': Ne_media assumida como Ne_baixa ({ne_final_media:.0f} cm⁻³)")
    
    if np.isnan(Te_nii) and np.isnan(Te_oiii):
        Te_nii = Te_oiii = 10000.0
        print(f"-> Fallback em '{obj_name}': Te assumida como 10000 K para ambas as zonas")
    elif np.isnan(Te_oiii):
        Te_oiii = Te_nii
        print(f"-> Fallback em '{obj_name}': Te_media assumida como Te_baixa ({Te_oiii:.0f} K)")
    elif np.isnan(Te_nii):
        Te_nii = Te_oiii
        print(f"-> Fallback em '{obj_name}': Te_baixa assumida como Te_media ({Te_nii:.0f} K)")
        
    Te_nii_final = np.clip(Te_nii, 6000, 20000)
    Te_oiii_final = np.clip(Te_oiii, 6000, 20000)
    
    return {'Te_baixa': Te_nii_final, 'Ne_baixa': ne_final_baixa, 'Te_media': Te_oiii_final, 'Ne_media': ne_final_media}

resultados_diagnostico = []
for obj, obs in obs_corrigidas.items():
    diagnostico = obter_diagnostico_robusto(obs, obj)
    diagnostico['Objeto'] = obj
    resultados_diagnostico.append(diagnostico)
df_diagnostico_exp = pd.DataFrame(resultados_diagnostico)
df_diagnostico_exp.to_csv("diagnostico_final_traduzido.csv", index=False)
print("-> Diagnóstico salvo em 'diagnostico_final_traduzido.csv'")

# ==============================================================================
# --- ETAPA 4: Cálculo de Abundâncias Iônicas (VERSÃO ROBUSTA) ---
# ==============================================================================
print("\n--- Etapa 4: Calculando abundâncias iônicas por objeto ---")

IONS_POR_ZONA = {
    'baixa': {
        'H+':  {'atom': pn.RecAtom('H', 1), 'lines': ['H1r_3889A', 'H1r_3970A', 'H1r_4102A', 'H1r_4341A', 'H1r_4861A', 'H1r_6563A']},
        'N0':  {'atom': pn.Atom('N', 1), 'lines': ['N1_5200A']},
        'N+':  {'atom': pn.Atom('N', 2), 'lines': ['N2_5755A', 'N2_6548A', 'N2_6584A']},
        'O+':  {'atom': pn.Atom('O', 2), 'lines': ['O2_3726A', 'O2_3729A', 'O2_7319A', 'O2_7330A']},
        'O0':  {'atom': pn.Atom('O', 1), 'lines': ['O1_6300A', 'O1_6364A']},
        'S+':  {'atom': pn.Atom('S', 2), 'lines': ['S2_4069A', 'S2_4076A', 'S2_6716A', 'S2_6731A']},
    },
    'media': {
        'He+':    {'atom': pn.RecAtom('He', 1), 'lines': ['He1r_4471A', 'He1r_5876A', 'He1r_6678A', 'He1r_7065A']},
        'He++':   {'atom': pn.RecAtom('He', 2), 'lines': ['He2r_4686A', 'He2r_5411A']},
        'O++':    {'atom': pn.Atom('O', 3), 'lines': ['O3_4363A', 'O3_4959A', 'O3_5007A']},
        'S++':    {'atom': pn.Atom('S', 3), 'lines': ['S3_6312A', 'S3_9069A', 'S3_9532A']},
        'Ar++':   {'atom': pn.Atom('Ar', 3), 'lines': ['Ar3_7136A', 'Ar3_7751A']},
        'Ar+++':  {'atom': pn.Atom('Ar', 4), 'lines': ['Ar4_4711A', 'Ar4_4740A', 'Ar4_7170A', 'Ar4_7237A', 'Ar4_7263A']},
        'Ar++++': {'atom': pn.Atom('Ar', 5), 'lines': ['Ar5_6435A', 'Ar5_7005A']},
        'Ne++':   {'atom': pn.Atom('Ne', 3), 'lines': ['Ne3_3869A', 'Ne3_3968A']},
        'Cl++':   {'atom': pn.Atom('Cl', 3), 'lines': ['Cl3_5518A', 'Cl3_5538A']},
        'Cl+++':  {'atom': pn.Atom('Cl', 4), 'lines': ['Cl4_8046A']}, # Adicionei um exemplo de linha para Cl+++
        'Ca++++': {'atom': pn.Atom('Ca', 5), 'lines': ['Ca5_5309A']},
    }
}

resultados_por_objeto = []
for _, linha_diagnostico in df_diagnostico_exp.iterrows():
    obj_id = linha_diagnostico['Objeto']
    obs = obs_corrigidas.get(obj_id)
    if not obs: continue

    print(f"Processando objeto: {obj_id}")
    abunds_obj = {'Objeto': obj_id}

    Te_baixa, Ne_baixa = linha_diagnostico['Te_baixa'], linha_diagnostico['Ne_baixa']
    Te_media, Ne_media = linha_diagnostico['Te_media'], linha_diagnostico['Ne_media']

    # --- ZONAS BAIXA E MÉDIA ---
    for zona, ions_na_zona in IONS_POR_ZONA.items():
        Te = Te_baixa if zona == 'baixa' else Te_media
        Ne = Ne_baixa if zona == 'baixa' else Ne_media
        
        if pd.notna(Te) and pd.notna(Ne):
            for ion, data in ions_na_zona.items():
                abunds_individuais, pesos_fluxo = [], []
                for label in data['lines']:
                    line = next((l for l in obs.lines if l.label == label), None)
                    if line and pd.notna(line.corrIntens) and line.corrIntens > 0:
                        try:
                            abund = data['atom'].getIonAbundance(line.corrIntens, tem=Te, den=Ne, wave=line.wave)
                            abunds_individuais.append(abund)
                            pesos_fluxo.append(line.corrIntens)
                        except Exception:
                            pass
                if abunds_individuais:
                    abunds_obj[ion] = np.average(abunds_individuais, weights=pesos_fluxo)

    resultados_por_objeto.append(abunds_obj)

# --- Criar e salvar o DataFrame final ---
df_final_real = pd.DataFrame(resultados_por_objeto)

# Reorganiza as colunas
cols_ordem = ['Objeto', 'H+', 'He+', 'He++']
outras_cols = sorted([c for c in df_final_real.columns if c not in cols_ordem])
df_final_real = df_final_real[cols_ordem + outras_cols]

# Salva a tabela REAL (com possíveis NaNs)
df_final_real.to_csv("abundancias_por_objeto_REAL.csv", index=False, float_format='%.4e')

# Cria e salva a tabela LOG
df_final_log = df_final_real.copy()
for col in df_final_log.columns:
    if col != 'Objeto':
        df_final_log[col] = df_final_real[col].apply(lambda x: 12 + np.log10(x) if pd.notna(x) and x > 0 else np.nan)
df_final_log.to_csv("abundancias_por_objeto_LOG.csv", index=False, float_format='%.4f')

print("\n-> Tabelas de abundâncias por OBJETO (REAL e LOG) salvas com sucesso!")


# ==================================================================================
# --- ETAPA 5: CÁLCULO DE ICFS E ABUNDÂNCIAS (COM CORREÇÃO MANUAL E NOVA FÓRMULA N) ---
# ==================================================================================
print("\n--- Etapa 5: Calculando abundâncias elementares ---")

icf = pn.ICF()
#ICFs usados: {He('KH01_4a'), Ar('direct_Ar.345'), 
#                Cl('KH01_4f'), N('KB94_A0'), Ne('KH01_4d'), 
#                  O('KH01_4b'), S('direct_S.234')}
print(icf.getExpression('KH01_4a')) 

print(icf.getExpression('direct_Ar.345')) 

print(icf.getExpression('KH01_4f')) 

print(icf.getExpression('KB94_A0')) 

print(icf.getExpression('KH01_4d')) 

print(icf.getExpression('KH01_4b')) 

print(icf.getExpression('direct_S.234'))

try:
    df_ions = pd.read_csv("abundancias_por_objeto_REAL.csv")
except FileNotFoundError:
    print("-> ERRO CRÍTICO: 'abundancias_por_objeto_REAL.csv' não encontrado. Execute a Etapa 4 primeiro.")
    exit()

# --- PREPARAÇÃO E CORREÇÃO MANUAL DOS DADOS ---
# Substitui todos os valores 'NaN' (ausentes) do CSV por 0.0
df_ions.fillna(0.0, inplace=True)
print("-> Correção: Todos os valores de abundância iônica ausentes foram substituídos por 0.0.")

# Garante que as colunas para a nova fórmula de Nitrogênio existam
# Se não existirem (o que é o caso), elas são criadas com valor zero.
required_n_ions = ['N0', 'N+']
for ion in required_n_ions:
    if ion not in df_ions.columns:
        print(f"-> Aviso: Coluna '{ion}' (necessária para nova fórmula de N) não encontrada. Será tratada como zero.")
        df_ions[ion] = 0.0

resultados_elementares = []

for index, objeto_ions in df_ions.iterrows():
    resultado_obj = {'Objeto': objeto_ions['Objeto']}
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # --- RESOLVENDO O "SISTEMA DE EQUAÇÕES" PARA CADA OBJETO ---
        
        He = objeto_ions['He+'] + objeto_ions['He++']
        icf_he = He / objeto_ions['He+']
        O = (objeto_ions['O+'] + objeto_ions['O++']) * icf_he
 
        N = objeto_ions['N0'] + objeto_ions['N+'] 
        
        Ne = (objeto_ions['Ne++'] / objeto_ions['O++']) * (objeto_ions['O+'] + objeto_ions['O++']) * icf_he
       
        S = objeto_ions['S+'] + objeto_ions['S++'] 
        
        # Usamos .get() por segurança, caso a coluna Cl+++ não seja criada na Etapa 4
        Cl = (objeto_ions['Cl++'] + objeto_ions.get('Cl+++', 0.0)) * icf_he
        
        # A fórmula do Argônio depende da nova fórmula do Nitrogênio
        Ar = objeto_ions['Ar++'] + objeto_ions['Ar+++'] + objeto_ions['Ar++++']
        resultado_obj.update({'He/H': He, 'O/H': O, 'N/H': N, 'Ne/H': Ne, 'S/H': S, 'Cl/H': Cl, 'Ar/H': Ar})
        
    resultados_elementares.append(resultado_obj)

# --- Montar, limpar e salvar os DataFrames ELEMENTARES ---
df_elemental_real = pd.DataFrame(resultados_elementares)
df_elemental_real.replace([np.inf, -np.inf], np.nan, inplace=True)
df_elemental_real.to_csv("abundancias_elementares_REAL.csv", index=False, float_format='%.4e')

df_elemental_log = df_elemental_real.copy()
for col in df_elemental_log.columns:
    if col != 'Objeto':
        df_elemental_log[col] = df_elemental_real[col].apply(lambda x: 12 + np.log10(x) if pd.notna(x) and x > 0 else np.nan)
df_elemental_log.to_csv("abundancias_elementares_LOG.csv", index=False, float_format='%.4f')

print("\n-> Tabelas de abundâncias ELEMENTARES (REAL e LOG) salvas com sucesso!")
print("\nPré-visualização dos resultados (abundâncias elementares reais):")
print(df_elemental_real.head())

# ==================================================================================
# --- ETAPA 6:  MÉDIAS DAS EXPOSIÇÕES ---
# ==================================================================================
print("\n--- Etapa 6:   Médias das exposições   ---")

def consolidar_resultados(caminho_arquivo_entrada, caminho_arquivo_saida):
    """
    Lê um arquivo de resultados (iônico ou elementar), agrupa por objeto base,
    calcula a média e o desvio padrão, e salva um novo arquivo consolidado.
    """
    try:
        df = pd.read_csv(caminho_arquivo_entrada)
    except FileNotFoundError:
        print(f"-> Aviso: Arquivo '{caminho_arquivo_entrada}' não encontrado. Pulando esta consolidação.")
        return

    # 1. Extrai o nome base do objeto para o agrupamento.
    # Ex: 'MaC_1-16_1' -> 'MaC_1-16'
    # A lógica verifica se a última parte após '_' é um número, para não agrupar errado.
    df['Objeto_Base'] = df['Objeto'].apply(
        lambda x: x.rsplit('_', 1)[0] if x.rsplit('_', 1)[-1].isdigit() else x
    )

    # 2. Seleciona apenas as colunas numéricas para os cálculos.
    colunas_numericas = df.select_dtypes(include=np.number).columns.tolist()

    # 3. Agrupa pelo nome base e calcula a média e o desvio padrão (std).
    # O .agg() permite aplicar múltiplas funções de uma só vez.
    df_agrupado = df.groupby('Objeto_Base')[colunas_numericas].agg(['mean', 'std'])

    # 4. 'Achata' os MultiIndex das colunas para um formato mais simples.
    # Ex: a coluna ('O/H', 'mean') vira 'O/H_mean'
    df_agrupado.columns = ['_'.join(col).strip() for col in df_agrupado.columns.values]
    df_agrupado.reset_index(inplace=True)

    # Salva o DataFrame final e consolidado.
    df_agrupado.to_csv(caminho_arquivo_saida, index=False, float_format='%.4f')
    print(f"-> Resultados consolidados salvos em: '{caminho_arquivo_saida}'")
    
    return df_agrupado

# --- Consolida os resultados IÔNICOS e ELEMENTARES ---
df_ionico_final = consolidar_resultados(
    caminho_arquivo_entrada="abundancias_por_objeto_LOG.csv",
    caminho_arquivo_saida="abundancias_ionicas_CONSOLIDADO.csv"
)

df_elemental_final = consolidar_resultados(
    caminho_arquivo_entrada="abundancias_elementares_LOG.csv",
    caminho_arquivo_saida="abundancias_elementares_CONSOLIDADO.csv"
)

# --- Exibe uma pré-visualização dos resultados elementares consolidados ---
if df_elemental_final is not None:
    print("\nPré-visualização dos resultados elementares consolidados (média e desvio padrão):")
    print(df_elemental_final.head())

print("\n--- Script finalizado com sucesso! ---")
