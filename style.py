#figure style for ieee mlsp, wong 2011 palette
import matplotlib.pyplot as plt

COL1 = 3.39   #single column (86mm)
COL2 = 7.0    #full textwidth (178mm)

PAL = {
    'raw':       '#888888',
    'sg':        '#E69F00',
    'sg_sel':    '#56B4E9',
    'specparam': '#009E73',
    'svd':       '#B8860B',
    'sccd':      '#0072B2',
    'qppca':     '#D55E00',
    'qmf':       '#CC79A7',
    'dae':       '#5E2A8A',
    'text':      '#1a1a1a',
    'neutral':   '#666666',
    'identity':  '#2D3142',
    'truth':     '#333333',
}
LABELS = {
    'raw': 'Raw', 'sg': 'SG', 'sg_sel': 'SG-sel', 'specparam': 'Specparam',
    'svd': 'SVD', 'sccd': 'SCCD', 'qppca': 'Q-PPCA', 'qmf': 'QMF', 'dae': 'DAE',
}
ORDER = ['raw', 'sg', 'sg_sel', 'svd', 'sccd', 'qppca', 'qmf', 'dae']


def set_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'axes.titleweight': 'normal',
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,
        'legend.fontsize': 7,
        'legend.frameon': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'lines.linewidth': 1.0,
        'axes.grid': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.04,
        #42 keeps fonts as truetype so the pdf stays editable
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
    })


def panel_label(ax, label, x=-0.12, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=11,
            fontweight='bold', color=PAL['text'], va='top')


def save_fig(fig, stem):
    for ext in ('svg', 'pdf', 'png'):
        fig.savefig(f'{stem}.{ext}')
    plt.close(fig)
    print(f'  saved {stem}')
