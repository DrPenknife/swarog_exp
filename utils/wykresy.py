%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import numpy as np
from matplotlib import rcParams
rcParams["font.family"] = "monospace"
# Dane
ds1 = pd.DataFrame({
    'group': ['Baseline', 'CNN', 'LSTM', 'BERT-1', 'BERT-2'],
    'Precision': [2.61, 3.33, 3.36, 4.19, 3.16, 5.19, 6.16],
    'Recall': [6.04, 4.57, 5.37, 3.4, 4.39, 2.82, 1.4],
    'Specificity': [1.56, 3.45, 2.53, 4.62, 3.54, 5.56, 6.74],
    'AUC': [5.46, 4.46, 5.15, 3.47, 4.14, 2.96, 2.35],
    'F-measure': [4.15, 3.68, 4.24, 3.4, 3.54, 3.91, 5.07],
    'G-mean': [5.78, 4.38, 5.39, 3.46, 4.13, 2.81, 2.05],
})
 
dfs = [CART, KNN, LSVM, RSVM]
# tytuły i nazwy plików
names = ["CART", "KNN", "L-SVM", "R-SVM"]
# kolory
colors = [(0, 0, 0), (0, 0, 0.9), (0.9, 0, 0), (0.9, 0, 0), (0, 0, 0.9), (0, 0, 0.9), (0.9, 0, 0)]
# styl linii
ls = ["-", "-", "-", "--", "--", ":", ":"]
# grubosc linii
lw = [1, 1, 1, 1, 1, 1, 1]
for i, title in enumerate(names):
    # number of variable
    df = dfs[i]
    categories = list(df)[1:]
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    # No shitty border
    ax.spines["polar"].set_visible(False)
    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles, categories)
    # Adding plots
    for i in range(7):
        values = df.loc[i].drop("group").values.flatten().tolist()
        values += values[:1]
        #print(values)
        values = [float(i) for i in values]
        ax.plot(
            angles, values, label=df.iloc[i, 0], c=colors[i], ls=ls[i], lw=lw[i],
        )
    # Add legend
    plt.legend(
        loc="lower center",
        ncol=4,
        columnspacing=1,
        frameon=False,
        bbox_to_anchor=(0.5, -0.3),
        fontsize=9,
    )
    # Add a grid
    plt.grid(ls=":", c=(0.7, 0.7, 0.7))
    # Add a title
    plt.title("%s" % (title), size=8, y=1.08, fontfamily="serif")
    #plt.tight_layout()
    # Draw labels
    a = np.linspace(0, 1, 8)
    plt.yticks(
            [0,1, 2, 3, 4, 5, 6, 7],
            ["0", "1", "2", "3", "4", "5", "6", "7"],
            fontsize=9,
        )
    plt.ylim(0.0, 7.0)
    plt.gcf().set_size_inches(4, 3.5)
    plt.gcf().canvas.draw()
    angles = np.rad2deg(angles)
    ax.set_rlabel_position((angles[0] + angles[1]) / 2)
    har = [(a >= 90) * (a <= 270) for a in angles]
    for z, (label, angle) in enumerate(zip(ax.get_xticklabels(), angles)):
        x, y = label.get_position()
        #print(label, angle)
        lab = ax.text(
            x, y, label.get_text(), transform=label.get_transform(), fontsize=9,
        )
        lab.set_rotation(angle)
        if har[z]:
            lab.set_rotation(180 - angle)
        else:
            lab.set_rotation(-angle)
        lab.set_verticalalignment("center")
        lab.set_horizontalalignment("center")
        lab.set_rotation_mode("anchor")
    for z, (label, angle) in enumerate(zip(ax.get_yticklabels(), a)):
        x, y = label.get_position()
        #print(label, angle)
        lab = ax.text(
            x,
            y,
            label.get_text(),
            transform=label.get_transform(),
            fontsize=6,
            #c=(0.7, 0.7, 0.7),
        )
        lab.set_rotation(-(angles[0] + angles[1]) / 2)
        lab.set_verticalalignment("bottom")
        lab.set_horizontalalignment("center")
        lab.set_rotation_mode("anchor")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    #plt.savefig("%s.png" % (title), bbox_inches='tight', dpi=300)
    #plt.close()
    plt.show()