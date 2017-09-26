from matplotlib.font_manager import FontProperties
import pylab as plt
import numpy as np
import pickle

def plot_periodic_table(vals_dict, title, cmap='YlGn', 
        vmin = 0, vmax = 1.5, show=True, filename=None,
        thresh = None, num_format='%3.2f'):
        if not thresh:
                thresh = 2.*(vmax-vmin)/3
        # Only supports up to atomic number 112
        positions = {
        'H': (0, 0), 'He': (0, 17),
        'Li': (1, 0), 'Be': (1, 1),
        'B': (1, 12), 'C': (1, 13), 'N': (1, 14), 'O': (1, 15), 'F': (1, 16), 'Ne': (1, 17),
        'Na': (2, 0), 'Mg': (2, 1),
        'Al': (2, 12), 'Si': (2, 13), 'P': (2, 14), 'S': (2, 15), 'Cl': (2, 16), 'Ar': (2, 17),
        'K': (3, 0), 'Ca': (3, 1),
        'Sc': (3, 2), 'Ti': (3, 3), 'V': (3, 4), 'Cr': (3, 5), 'Mn': (3, 6), 
                'Fe': (3, 7), 'Co': (3, 8), 'Ni': (3, 9), 'Cu': (3, 10), 'Zn': (3, 11),
        'Ga': (3, 12), 'Ge': (3, 13), 'As': (3, 14), 'Se': (3, 15), 'Br': (3,16), 'Kr': (3, 17),
        'Rb': (4, 0), 'Sr': (4, 1), 
        'Y': (4, 2), 'Zr': (4, 3), 'Nb': (4, 4), 'Mo': (4, 5), 'Tc': (4, 6),
                'Ru': (4, 7), 'Rh': (4, 8), 'Pd': (4, 9), 'Ag': (4, 10), 'Cd': (4,11),
        'In': (4, 12), 'Sn': (4, 13), 'Sb': (4, 14), 'Te': (4, 15), 'I': (4, 16), 'Xe': (4, 17),
        'Cs': (5, 0), 'Ba': (5, 1),
        'Hf': (5, 3), 'Ta': (5, 4), 'W': (5, 5), 'Re': (5, 6),
                'Os': (5, 7), 'Ir': (5, 8), 'Pt': (5, 9), 'Au': (5, 10), 'Hg': (5, 11),
        'Tl': (5, 12), 'Pb': (5, 13), 'Bi': (5, 14), 'Po': (5, 15), 'At': (5, 16), 'Rn': (5, 17),
        'Fr': (6, 0), 'Ra': (6, 1),
        'Rf': (6, 3), 'Db': (6, 4), 'Sg': (6, 5), 'Bh': (6, 6),
                'Hs': (6, 7), 'Mt': (6, 8), 'Ds': (6, 9), 'Rg': (6, 10), 'Cn': (6, 11),
        'La': (7, 3), 'Ce': (7, 4), 'Pr': (7, 5), 'Nd': (7, 6), 'Pm': (7, 7), 
                'Sm': (7, 8), 'Eu': (7, 9), 'Gd': (7, 10), 'Tb': (7, 11), 'Dy': (7, 12),
                'Ho': (7, 13), 'Er': (7, 14), 'Tm': (7, 15), 'Yb': (7, 16), 'Lu': (7,17),
        'Ac': (8, 3), 'Th': (8, 4), 'Pa': (8, 5), 'U': (8, 6), 'Np': (8, 7), 
                'Pu': (8, 8), 'Am': (8, 9), 'Cm': (8, 10), 'Bk': (8, 11), 'Cf': (8, 12),
                'Es': (8, 13), 'Fm': (8, 14), 'Md': (8, 15), 'No': (8, 16), 'Lr': (8, 17)
        }

        mat = np.nan * np.ones(shape=(9,18))

        plt.figure(figsize=(16,8))
        font = FontProperties()
        font.set_weight('bold')
        for key in vals_dict.keys():
                pos = positions[key]
                x = pos[0]
                y = pos[1]
                rmse = vals_dict[key]
                mat[x][y] = rmse
                plt.plot([y-0.5,y+0.5],[x-0.5,x-0.5],lw=3,color='black')
                plt.plot([y-0.5,y+0.5],[x+0.5,x+0.5],lw=3,color='black')
                plt.plot([y+0.5,y+0.5],[x-0.5,x+0.5],lw=3,color='black')
                plt.plot([y-0.5,y-0.5],[x-0.5,x+0.5],lw=3,color='black')
                if rmse > thresh:
                        color='white'
                else:
                        color='black'
                plt.text(x=y, y=x-0.1, 
                        s=key, fontsize=14,
                        horizontalalignment='center',
                        fontproperties=font,
                        color=color)
                plt.text(x=y, y=x+0.2,
                        s=num_format%rmse, fontsize=14,
                        horizontalalignment='center',
                        fontproperties=font,
                        color=color)

        plt.imshow(mat,
                interpolation='none',
                cmap=cmap,
                vmin = vmin,
                vmax = vmax)
        plt.xticks([])
        plt.yticks([])
        plt.title(title, size=18)
        plt.tight_layout()
        if show:
                plt.show()
        if filename:
                plt.savefig(filename)

