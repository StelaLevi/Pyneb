#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 15:46:36 2025

@author: stela
"""

import pyneb as pn
import numpy as np
import matplotlib.pyplot as plt

obs = pn.Observation()
obs.readData('obs.dat', fileFormat='lines_in_rows', err_default=0.05) # fill obs with data read from observations1.dat
print("Linhas carregadas:")
print(obs.getSortedLines())

obs.def_EBV(label1="H1r_6563A", label2="H1r_4861A", r_theo=2.85)

print(obs.extinction.E_BV)
obs.extinction.law = 'CCM89'  # define the extinction law from Cardelli et al.
obs.correctData(normWave=4861.)


obs.printIntens()
#obs.printIntens(returnObs=True)

#for line in obs.getSortedLines(): 
#    print(line.label, line.corrIntens[0])

diags = pn.Diagnostics()
#diags.addDiagsFromObs(obs)
#diags.diags

diags.addDiag([
              '[NII] 5755/6584', 
              '[OIII] 4363/5007', 
              '[SII] 6731/6716',
              '[ArIV] 4740/4711',
              '[ClIII] 5538/5518',
              '[ArIV] 7230+/4720+'
              ])

#diags.eval_diag('[NII] 5755/6548')
# %config InlineBackend.figure_format = 'png'
# mpl.rc("savefig", dpi=150)
# emisgrids = pn.getEmisGridDict(atomDict=diags.atomDict)
# diags.plot(emisgrids, obs)

emisgrids = pn.getEmisGridDict(atom_list=diags.getUniqueAtoms(), den_max=1e6)
diags.plot(emisgrids, obs)


# Display the plot
plt.show()

#The observed ratio can be automatically extracted from an Observation object named obs:
Te, Ne = diags.getCrossTemDen('[NII] 5755/6548', '[SII] 6731/6716', obs=obs)
print('Te = {0:5.0f} K, Ne = {1:7.1f} cm-1'.format(Te, Ne))

# Atom creation and definition of physical conditions 
o3=pn.Atom('O', 3)
tem=np.arange(100)*300+300
den = Ne

# Comment the second if you want all the lines to be plotted
lineList=o3.lineList
lineList=[4363, 4959, 5007]

# Plot	
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim([1.e-30, 5e-20])

for line in lineList:
    y=o3.getEmissivity(tem, den, wave=line)
    plt.semilogy(tem, y,  label="{:.0f}".format(line))

plt.xlabel('T$_e$ [K]')
plt.ylabel("j(T) [erg cm$^{-3}$ s${-1}$]")
plt.legend(loc='lower right')
plt.title('[O III] emissivities @ N$_e$={:.0f}'.format(den))
plt.show()

obs = pn.Observation()
obs.readData('obs.dat', fileFormat='lines_in_rows', err_default=0.05) # fill obs with data read from observations1.dat
obs.def_EBV(label1="H1r_6563A", label2="H1r_4861A", r_theo=2.85)
obs.correctData(normWave=4861.)

