"""
code to generate the image for the color bar for case studies in the paper.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

matplotlib.rcParams.update({'font.size': 16})
plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False

name = 'afmhot'
fwidth = 2
fhigh = 6
gradient = np.linspace(0, 1, 1000)
gradient = np.vstack((gradient, gradient))
gradient = np.transpose(gradient)

fig, ax = plt.subplots(nrows=1, figsize=(fwidth, fhigh))
plt.subplots_adjust(left=0.2, right=0.5)
#fig, ax = plt.subplots()

#ax.grid(False)
#labels = [item.get_text() for item in ax.get_yticklabels()]
labels = ['0.0','0.2','0.4','0.6','0.8','1.0']
ax.set_xticklabels([])
ax.set_yticks(np.array([0,200,400,600,800,999]))
ax.set_yticklabels(labels)
#plt.axis([0, 1, 0, 1])

ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name), origin='lower')
ax.axes.get_xaxis().set_visible(False)
#ax.axes.get_yaxis().set_visible(False)
#ax.set_frame_on(False)

plt.savefig('./color_scale.png', dpi=1200, format='png')
plt.show()