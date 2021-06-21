# pylint: disable=E0401  # E0401: import-error (torch & sklearn)
import os
from datetime import datetime
import pickle
from tqdm import tqdm
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from sklearn import metrics
from sklearn.model_selection import train_test_split

import torch
from torchvision.utils import make_grid

# ================================================================================================================================= #
#                                                             Time Tags                                                             #
# ================================================================================================================================= #

def now(abbv=False):
    n = datetime.now()
    return n.strftime("%m%d-%H%M%S") if abbv else n.strftime("%m%d-%H%M%S-%f")


def today():
    return datetime.now().strftime("%m%d")


# ================================================================================================================================= #
#                                                            Pickle File                                                            #
# ================================================================================================================================= #
def dump(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)


def load(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


# ================================================================================================================================= #
#                                                            Grid Images                                                            #
# ================================================================================================================================= #

def grid_image(image: torch.Tensor, padding=2, pad_value=0., in_3d=False):
    """
    get a grid of image (tensor) with each row showing the slices in one 3d image
    Args:
        image: Tensor of shape [n, num_slice, num_row, num_col]
        padding: number of pixels padding between images
        pad_value: padding color, 0.0 (black) to 1.0 (white)
    """
    if in_3d:
        image = image.squeeze(1)
    num_slice, num_row, num_col = image.shape[1:]
    return make_grid(image.view(-1, 1, num_row, num_col), nrow=num_slice, normalize=True, padding=padding, pad_value=pad_value)


# ================================================================================================================================= #
#                                                                ROC                                                                #
# ================================================================================================================================= #

def roc_curves(results, types=('fake', 'real', 'noise'), names=None, figsize=(10, 10)):
    if names is None:
        names = types
    fig, ax = plt.subplots(figsize=figsize)
    for t, n in zip(types, names):
        df = results[results.type == t]
        fpr, tpr, _ = metrics.roc_curve(df.label, df.soft, pos_label=1)
        ax.plot(fpr, tpr, lw=2, label=f"{n} (area = {metrics.auc(fpr, tpr):.3f})")
    ax.legend(fontsize=20)
    ax.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    ax.set_title("ROC curves", fontsize=24)
    ax.set_xlabel("False Positive Rate", fontsize=16)
    ax.set_ylabel("True Positive Rate", fontsize=16)
    return fig


def roc_test(results, types=('fake', 'real'),
             roc_boot_file=None, ignore_exists=False,  # store/load bootstrapped distribution to save time
             boot_n=2000, disable_progress_bar=False):
    # pylint: disable=W0511  # fix-me issue (todos)
    """
    ROC and AUC (test): compare roc (results['soft'] vs results['label'])  # todo make input more general

    (R implementation (pROC/roc.test) not easy to import under BUSDEV  => implement python version myself)
    ref: https://www.rdocumentation.org/packages/pROC/versions/1.16.2/topics/roc.test
    paper: Venkatraman, E. S. "A permutation test to compare receiver operating characteristic curves." Biometrics 56.4 (2000): 1134-1138.
        https://pubmed.ncbi.nlm.nih.gov/11129471/
    """
    df1, df2 = results[results.type == types[0]], results[results.type == types[1]]
    df1['r'], df2['r'] = df1.soft.rank(method='first'), df2.soft.rank(method='first')
    r1, r2 = np.array(df1.r), np.array(df2.r)
    d1, d2 = np.array(df1.label), np.array(df2.label)
    kappa = (np.sum(d1) + np.sum(d2)) / (d1.shape[0] + d2.shape[0])

    def roc_stat(_r1, _r2, _d1, _d2):
        n, m = _d1.shape[0], _d2.shape[0]
        r10, r11 = _r1[_d1 == 0], _r1[_d1 == 1]
        r20, r21 = _r2[_d2 == 0], _r2[_d2 == 1]

        f1 = np.array([np.mean(r11 <= i+1) for i in range(n)])
        g1 = np.array([np.mean(r10 <= i+1) for i in range(n)])
        f2 = np.array([np.mean(r21 <= i+1) for i in range(m)])
        g2 = np.array([np.mean(r20 <= i+1) for i in range(m)])

        p1, p2 = kappa * f1 + (1 - kappa) * g1, kappa * f2 + (1 - kappa) * g2
        exp1, exp2 = kappa * f1 + (1 - kappa) * (1 - g1), kappa * f2 + (1 - kappa) * (1 - g2)

        x = np.hstack([p1, p2])
        fx1, fx2 = np.interp(x, p1, exp1), np.interp(x, p2, exp2)

        # trapezoid integration
        return np.trapz(np.abs(fx1 - fx2), x)

    def permute():
        r10, r11 = r1[d1 == 0], r1[d1 == 1]
        r20, r21 = r2[d2 == 0], r2[d2 == 1]
        _r10, _r20 = train_test_split(np.hstack([r10, r20]), train_size=r10.shape[0])
        _r11, _r21 = train_test_split(np.hstack([r11, r21]), train_size=r11.shape[0])
        _r1, _r2 = np.zeros(r1.shape), np.zeros(r2.shape)
        _r1[d1 == 0] = _r10
        _r2[d2 == 0] = _r20
        _r1[d1 == 1] = _r11
        _r2[d2 == 1] = _r21
        return _r1, _r2, d1, d2

    if roc_boot_file is not None and os.path.exists(roc_boot_file) and not ignore_exists:
        s, ss = load(roc_boot_file)
    else:
        s = roc_stat(r1, r2, d1, d2)
        ss = np.array([roc_stat(*permute()) for _ in tqdm(range(boot_n), desc="roc test", disable=disable_progress_bar)])
        if roc_boot_file is not None:
            dump((s, ss), roc_boot_file)

    pval = np.mean(ss > s)

    print("Venkatraman's test for two unpaired ROC curves: \n"
          f"test statistics = {s:.4f}, p-value = {pval:.2f} {'>' if pval > 0.05 else '<'} 0.05 \n"
          f"==> There is {'*NO*' if pval > 0.05 else 'a'} significant difference between ROC curves. ")

    return s, pval
