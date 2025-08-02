

import pandas as pd
import numpy as np
import pyneb as pn
import os
from collections import defaultdict
import matplotlib.pyplot as plt


# --- Etapa 1: Preparando os Dados Brutos ---
print("--- Etapa 1: Lendo e normalizando os fluxos pelo H-beta ---")
try:
    # Carrega a planilha com todos os fluxos observados
    df_bruto = pd.read_excel('dados_A.ods', engine='odf')
except FileNotFoundError:
    print("\n!! ERRO: O arquivo 'dados_A.ods' não foi encontrado.")
    print("!! Por favor, coloque-o na mesma pasta do script antes de continuar.")
    exit()

# Pega o nome de cada coluna de objeto (exposição)
objetos = df_bruto.columns[1:]

# Cria um diretório para salvar os arquivos de fluxo normalizados
output_dir = 'dados_normalizados'
os.makedirs(output_dir, exist_ok=True)

# Dicionário para guardar o caminho de cada novo arquivo .dat
arquivos_dat = {}
for obj in objetos:
    try:
        # Encontra o fluxo de H-beta (H1r_4861A) para usar como referência
        linha_hbeta = df_bruto['LINE'] == 'H1r_4861A'
        if not linha_hbeta.any():
            raise ValueError("Linha de referência H1r_4861A não encontrada na planilha.")
        
        fluxo_hbeta = df_bruto.loc[linha_hbeta, obj].values[0]
        if pd.isna(fluxo_hbeta) or fluxo_hbeta == 0:
            raise ValueError(f"Fluxo de H-beta é inválido (zero ou NaN).")

        # Normaliza todos os fluxos da exposição pelo fluxo de H-beta
        df_norm = df_bruto[['LINE', obj]].copy()
        df_norm.columns = ['LINE', 'FLUX']
        df_norm['FLUX'] = pd.to_numeric(df_norm['FLUX'], errors='coerce').fillna(0) / fluxo_hbeta
        df_norm = df_norm[df_norm['FLUX'] > 0] # Remove linhas sem fluxo

        # Salva o arquivo .dat formatado para o PyNeb
        caminho_dat = os.path.join(output_dir, f'{obj}.dat')
        df_norm.to_csv(caminho_dat, sep='\t', index=False)
        arquivos_dat[obj] = caminho_dat

    except Exception as e:
        print(f"-> Aviso: Falha ao processar a exposição '{obj}': {e}")


# --- Etapa 2: Correção por Avermelhamento ---
print("\n--- Etapa 2: Corrigindo os fluxos pelo avermelhamento (extinção) ---")
obs_corrigidas = {}
for obj, caminho in arquivos_dat.items():
    try:
        # Cria um objeto Observation do PyNeb
        obs = pn.Observation()
        obs.readData(caminho, fileFormat='lines_in_rows', err_default=0.05)

        # Usa a razão H-alfa/H-beta para calcular o E(B-V)
        obs.def_EBV(label1="H1r_6563A", label2="H1r_4861A", r_theo=2.85)
        obs.extinction.law = 'F99'  # Lei de Fitzpatrick 1999
        obs.correctData(normWave=4861)

        # Guarda o objeto Observation com os dados já corrigidos
        obs_corrigidas[obj] = obs
    except Exception as e:
        print(f"-> Aviso: Falha ao corrigir a exposição '{obj}': {e}")


# --- Etapa 3: Diagnóstico de Te e Ne (por exposição) ---
print("\n--- Etapa 3: Calculando Temperatura e Densidade para cada exposição ---")
diags = pn.Diagnostics()
diags.addDiag(['[NII] 5755/6584', '[OIII] 4363/5007', '[SII] 6731/6716', '[ArIV] 4740/4711', '[ClIII] 5538/5518'])

for obj, obs in obs_corrigidas.items():
    try:
        print(f"\n Mostrando diagrama de diagnóstico para {obj}...")

        emisgrids = pn.getEmisGridDict(atom_list=diags.getUniqueAtoms(), den_max=1e6)

        diags.plot(emisgrids, obs, i_obs=0)
        plt.title(f'Diagrama de Diagnóstico – {obj}')
        plt.tight_layout()
        plt.show()  # Exibe na tela do Spyder

    except Exception as e:
        print(f" Erro ao gerar diagrama para {obj}: {e}")
# Pares de diagnósticos para cada zona de ionização
diagnosticos_por_zona = {
    'baixa': [('[NII] 5755/6584', '[SII] 6731/6716')],
    'média': [('[OIII] 4363/5007', '[ClIII] 5538/5518'), ('[OIII] 4363/5007', '[ArIV] 4740/4711')],
}

resultados_diagnostico = []
for obj, obs in obs_corrigidas.items():
    dados_exp = {'Objeto': obj, 'Te (baixa)': np.nan, 'Ne (baixa)': np.nan, 'Te (média)': np.nan, 'Ne_Cl (média)': np.nan, 'Ne_Ar (média)': np.nan}
    for zona, pares in diagnosticos_por_zona.items():
        for diag_tem, diag_den in pares:
            try:
                Te, Ne = diags.getCrossTemDen(diag_tem=diag_tem, diag_den=diag_den, obs=obs)
                if np.isnan(Te) or np.isnan(Ne): continue

                if zona == 'baixa':
                    dados_exp['Te (baixa)'] = Te
                    dados_exp['Ne (baixa)'] = Ne
                elif zona == 'média':
                    dados_exp['Te (média)'] = Te
                    if 'ClIII' in diag_den: dados_exp['Ne_Cl (média)'] = Ne
                    elif 'ArIV' in diag_den: dados_exp['Ne_Ar (média)'] = Ne
            except Exception:
                pass # Ignora se o cálculo falhar para um par específico
    resultados_diagnostico.append(dados_exp)

# Salva a tabela com os resultados por exposição
df_diagnostico_exp = pd.DataFrame(resultados_diagnostico)


# --- Etapa 4: Médias Ponderadas de Te e Ne (por nebulosa) ---
print("\n--- Etapa 4: Agrupando por nebulosa e calculando médias ponderadas de Te e Ne ---")
# Pesos para cada parâmetro (fluxos das linhas mais relevantes)
fluxos_para_peso = {
    'Te (baixa)': ['N2_5755A'], 'Ne (baixa)': ['S2_6716A', 'S2_6731A'],
    'Te (média)': ['O3_5007A'], 'Ne_Cl (média)': ['Cl3_5538A'], 'Ne_Ar (média)': ['Ar4_4740A']
}

def agrupar_por_nebulosa(nome_exposicao):
    return nome_exposicao.rsplit('_', 1)[0]

# Cria os grupos de exposições para cada nebulosa
grupos_neb = {base: [] for base in sorted(list(set(agrupar_por_nebulosa(obj) for obj in df_diagnostico_exp['Objeto'])))}
for obj in df_diagnostico_exp['Objeto']:
    grupos_neb[agrupar_por_nebulosa(obj)].append(obj)

dados_medias = defaultdict(list)
for neb, exposicoes in grupos_neb.items():
    dados_medias['Nebulosa'].append(neb)
    for param in ['Te (baixa)', 'Ne (baixa)', 'Te (média)', 'Ne_Cl (média)', 'Ne_Ar (média)']:
        valores, pesos = [], []
        for exp in exposicoes:
            valor_param = df_diagnostico_exp.loc[df_diagnostico_exp['Objeto'] == exp, param].values
            if not (valor_param.size > 0 and not pd.isna(valor_param[0])): continue

            fluxos_relacionados = [df_bruto.loc[df_bruto['LINE'] == l, exp].values[0] for l in fluxos_para_peso[param] if l in df_bruto['LINE'].values and not pd.isna(df_bruto.loc[df_bruto['LINE'] == l, exp].values[0])]
            if not fluxos_relacionados or sum(fluxos_relacionados) == 0: continue
            
            valores.append(valor_param[0])
            pesos.append(sum(fluxos_relacionados))

        if valores:
            dados_medias[param].append(np.average(valores, weights=pesos))
        else:
            dados_medias[param].append(np.nan)

df_medias_neb = pd.DataFrame(dados_medias)
df_medias_neb.to_csv("medias_ponderadas_nebulosas.csv", index=False, float_format='%.2f')
print("-> Arquivo 'medias_ponderadas_nebulosas.csv' salvo com sucesso.")


# --- Etapa 5: Abundâncias Iônicas (por exposição) ---
print("\n--- Etapa 5: Calculando abundâncias iônicas para cada exposição e zona ---")
# Lista completa de todas as linhas de interesse para o cálculo
linhas_alvo = ['Ne3_3869A', 'Ne3_3968A', 'O3_4363A', 'He1r_4471A', 'He2r_4686A', 'Ar4_4711A', 'Ar4_4740A', 'O3_4959A', 'O3_5007A', 'N1_5200A', 'Ca5_5309A', 'He2r_5411A', 'Cl3_5518A', 'Cl3_5538A', 'N2_5755A', 'He1r_5876A', 'O1_6300A', 'S3_6312A', 'O1_6364A', 'Ar5_6435A', 'N2_6548A', 'N2_6584A', 'He1r_6678A', 'S2_6716A', 'S2_6731A', 'Ar5_7005A', 'He1r_7065A', 'Ar3_7136A', 'Ar4_7170A', 'Ar4_7237A', 'Ar4_7263A', 'O2_7319A', 'Ar3_7751A']

resultados_abund_ionicas = {'baixa': defaultdict(dict), 'media_ClIII': defaultdict(dict), 'media_ArIV': defaultdict(dict)}

def calcular_abundancias_ionicas(obs, atom_dict, tem, den):
    abunds = {}
    if np.isnan(tem) or np.isnan(den): return abunds
    for line in obs.getSortedLines():
        if line.label not in linhas_alvo or line.atom in ['H1', 'He1', 'He2']: continue
        try:
            val = atom_dict[line.atom].getIonAbundance(int_ratio=line.corrIntens, tem=tem, den=den, to_eval=line.to_eval, Hbeta=1.0)
            if not np.isnan(val[0]) and val[0] > 0: abunds[line.label] = val[0]
        except Exception: pass
    return abunds

for _, linha_media in df_medias_neb.iterrows():
    nebulosa, exposicoes = linha_media['Nebulosa'], grupos_neb.get(linha_media['Nebulosa'], [])
    condicoes = {'baixa': (linha_media['Te (baixa)'], linha_media['Ne (baixa)']), 'media_ClIII': (linha_media['Te (média)'], linha_media['Ne_Cl (média)']), 'media_ArIV': (linha_media['Te (média)'], linha_media['Ne_Ar (média)'])}
    
    if not exposicoes or not obs_corrigidas.get(exposicoes[0]): continue
    atom_dict = pn.getAtomDict(atom_list=obs_corrigidas.get(exposicoes[0]).getUniqueAtoms())
    
    for exp in exposicoes:
        if not obs_corrigidas.get(exp): continue
        for zona, (Te, Ne) in condicoes.items():
            resultados_abund_ionicas[zona][exp].update(calcular_abundancias_ionicas(obs_corrigidas.get(exp), atom_dict, Te, Ne))

for zona, dados_zona in resultados_abund_ionicas.items():
    if not dados_zona: continue
    df_real = pd.DataFrame.from_dict(dados_zona, orient='index')
    df_real.index.name = 'Exposicao'
    for col in linhas_alvo:
        if col not in df_real.columns: df_real[col] = np.nan
    df_real = df_real.reindex(sorted(df_real.columns), axis=1)
    df_log = df_real.applymap(lambda x: 12 + np.log10(x) if pd.notnull(x) else np.nan)
    df_log.to_csv(f"abundancias_{zona}_LOG.csv")
print("-> Tabelas de abundâncias iônicas por exposição salvas.")


# --- Etapa 6: Médias Ponderadas das Abundâncias Iônicas ---
print("\n--- Etapa 6: Calculando médias ponderadas das abundâncias iônicas por nebulosa ---")
df_fluxos_raw = df_bruto.set_index('LINE')
for zona in ['baixa', 'media_ClIII', 'media_ArIV']:
    df_abundancias_log = pd.read_csv(f"abundancias_{zona}_LOG.csv")
    resultados_finais_zona = []
    for neb, exposicoes in grupos_neb.items():
        media_neb = {'Nebulosa': neb}
        df_abund_neb = df_abundancias_log[df_abundancias_log['Exposicao'].isin(exposicoes)]
        for ion in linhas_alvo:
            if ion not in df_abund_neb.columns: continue
            valores_abund, pesos_fluxo = [], []
            for exp in exposicoes:
                val_abund_series = df_abund_neb.loc[df_abund_neb['Exposicao'] == exp, ion]
                try: val_fluxo = df_fluxos_raw.loc[ion, exp]
                except KeyError: val_fluxo = np.nan
                if not val_abund_series.empty and pd.notna(val_abund_series.iloc[0]) and pd.notna(val_fluxo) and val_fluxo > 0:
                    valores_abund.append(val_abund_series.iloc[0]); pesos_fluxo.append(val_fluxo)
            if valores_abund: media_neb[ion] = np.average(valores_abund, weights=pesos_fluxo)
            else: media_neb[ion] = np.nan
        resultados_finais_zona.append(media_neb)
    df_medias_finais = pd.DataFrame(resultados_finais_zona)
    df_medias_finais.to_csv(f"medias_ponderadas_abundancias_{zona}.csv", index=False, float_format='%.4f')
print("-> Tabelas de médias ponderadas de abundâncias iônicas salvas.")


# --- Etapa 7: Abundâncias Elementais Finais (com ICF Manual) ---
print("\n--- Etapa 7: Calculando abundâncias elementais totais com Fatores de Correção de Ionização (ICF) ---")
print("-> Usando as fórmulas de Delgado-Inglada et al. (2014) implementadas manualmente.")

# Mapa de íons -> qual linha de emissão representa cada íon no cálculo
mapa_ions_para_icf = {
    'He+': 'He1r_5876A', 'He++': 'He2r_4686A',
    'O+': 'O1_6300A', 'O++': 'O3_5007A',
    'N+': 'N2_6584A',
    'S+': 'S2_6716A', 'S++': 'S3_6312A',
    'Ne++': 'Ne3_3869A',
    'Ar++': 'Ar3_7136A', 'Ar+++': 'Ar4_4740A'
}
elementos_alvo = ['He', 'O', 'N', 'S', 'Ne', 'Ar']

for zona in ['baixa', 'media_ClIII', 'media_ArIV']:
    try:
        df_abund_ion_medias = pd.read_csv(f"medias_ponderadas_abundancias_{zona}.csv").set_index('Nebulosa')
    except FileNotFoundError:
        print(f"\nAVISO: Arquivo 'medias_ponderadas_abundancias_{zona}.csv' não encontrado. Pulando zona '{zona}'.")
        continue

    print(f"\n>>> Processando zona de cálculo: {zona}")
    resultados_elementais_zona = []

    for nebulosa, linha_abund in df_abund_ion_medias.iterrows():
        dados_neb = {'Nebulosa': nebulosa}
        
        # Converte as abundâncias médias de log para valores lineares para os cálculos
        abunds = {}
        for ion_nome, col_df in mapa_ions_para_icf.items():
            if col_df in linha_abund and pd.notna(linha_abund[col_df]):
                abunds[ion_nome] = 10**(linha_abund[col_df] - 12)
        
        try:
            # Hélio: He/H = He+/H + He++/H (ICF é praticamente 1)
            if 'He+' in abunds and 'He++' in abunds:
                dados_neb['He'] = 12 + np.log10(abunds['He+'] + abunds['He++'])

            # Para os outros elementos, precisamos primeiro da abundância total de Oxigênio
            if 'O+' in abunds and 'O++' in abunds:
                O_total = abunds['O+'] + abunds['O++']
                dados_neb['O'] = 12 + np.log10(O_total)
                
                # Nitrogênio: N/H = (N+/H) * (O/O+). A correção é feita pela razão de O/O+.
                if 'N+' in abunds:
                    icf_N = O_total / abunds['O+']
                    N_total = abunds['N+'] * icf_N
                    dados_neb['N'] = 12 + np.log10(N_total)
                
                # Neônio: Ne/H = (Ne++/H) * (O/O++). Similar ao Nitrogênio.
                if 'Ne++' in abunds:
                    icf_Ne = O_total / abunds['O++']
                    Ne_total = abunds['Ne++'] * icf_Ne
                    dados_neb['Ne'] = 12 + np.log10(Ne_total)

            # Enxofre: A fórmula é um pouco mais complexa
            if 'S+' in abunds and 'S++' in abunds and 'O+' in abunds and 'O++' in abunds:
                O_total_S = abunds['O+'] + abunds['O++']
                razao_O_ionizado = abunds['O+'] / O_total_S
                icf_S = (1 - (1 - razao_O_ionizado)**3)**(-1/3.)
                S_total = (abunds['S+'] + abunds['S++']) * icf_S
                dados_neb['S'] = 12 + np.log10(S_total)

            # Argônio: Similar ao Enxofre, mas usa a razão de Nitrogênio como proxy
            if 'Ar++' in abunds and 'Ar+++' in abunds and 'N+' in abunds and 'O+' in abunds and 'O++' in abunds:
                O_total_Ar = abunds['O+'] + abunds['O++']
                N_total_Ar = abunds['N+'] * (O_total_Ar / abunds['O+'])
                razao_N_ionizado = abunds['N+'] / N_total_Ar
                icf_Ar = (1 - (1 - razao_N_ionizado)**3)**(-1/3.)
                Ar_total = (abunds['Ar++'] + abunds['Ar+++']) * icf_Ar
                dados_neb['Ar'] = 12 + np.log10(Ar_total)
                
        except (ValueError, KeyError, ZeroDivisionError):
            # Se qualquer cálculo falhar (ex: divisão por zero), o processo continua
            pass
            
        # Garante que todas as colunas de elementos existam no resultado final
        for elem in elementos_alvo:
            if elem not in dados_neb:
                dados_neb[elem] = np.nan
        
        resultados_elementais_zona.append(dados_neb)

    # Salva o resultado final da zona
    if resultados_elementais_zona:
        df_final_elemental = pd.DataFrame(resultados_elementais_zona)
        colunas_ordem = ['Nebulosa'] + elementos_alvo
        df_final_elemental = df_final_elemental[colunas_ordem]
        df_final_elemental.to_csv(f"abundancias_elementais_finais_{zona}.csv", index=False, float_format='%.4f')
        print(f"-> Tabela 'abundancias_elementais_finais_{zona}.csv' salva.")

print("\n\n--- TUDO PRONTO! Processo finalizado com sucesso. ---")