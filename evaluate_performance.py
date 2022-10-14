import numpy as np
from sklearn.metrics import auc, f1_score, precision_recall_curve


def num2val(nums, m):
    m = np.array(m)
    (n_rows, n_cols) = np.shape(m)
    nums = np.array(nums)
    col_ind = nums // n_rows
    row_ind = nums - n_rows * col_ind
    vals = []

    for i in range(len(nums)):
        if m[row_ind[i], col_ind[i]]:
            vals.append(1)
        else:
            vals.append(0)
    return vals


def evaluate_performance(class_score, label, alpha=3):
    """
    alpha: how many top pred to select to calculate f1
    """
    label = np.array(label) > 0
    (ncase, nclass) = np.shape(class_score)
    if nclass == 1:
        o = class_score > class_score.mean()
        pred = o*1
        acc = np.mean(pred == label)
    else:
        o = np.argsort(class_score * -1, axis=1)
        acc = np.mean(label[:, o[:, 0]])

    pred_ = class_score
    label_ = label

    # mi_aurpc = roc_auc_score(label_, pred_, average='micro')
    # Use AUC function to calculate the area under
    # the curve of precision recall curve

    precision, recall, thresholds = precision_recall_curve(
        label_.flatten(), pred_.flatten())
    f1_scores = 2*recall*precision/(recall+precision+1e-6)
    max_f1 = np.max(f1_scores)

    mi_auprc = auc(recall, precision)

    auprc = 0
    class_not_zero = 0
    for i in range(pred_.shape[1]):
        if label_[:, i].sum() != 0:
            precision, recall, thresholds = precision_recall_curve(
                label_[:, i], pred_[:, i])
            auprc += auc(recall, precision)
            class_not_zero += 1
    auprc /= class_not_zero
    ma_auprc = auprc

    return acc, max_f1, mi_auprc, ma_auprc


def evaluate_performance_new(pred, class_score, label):
    """
    alpha: how many top pred to select to calculate f1
    """
    label = np.array(label) > 0

    (ncase, nclass) = np.shape(class_score)

    acc = np.mean(pred * label)*nclass

    pred_flat = pred.copy()
    pred_flat = np.array(pred_flat.flatten(order="F"))
    label_flat = label.copy()
    label_flat = np.array(label_flat).flatten(order="F")
    label_flat = label_flat * 1
    # tab = crosstab(pred_flat, label_flat)
    # f1 = 2 * tab[1][1] / (2 * tab[1][1] + tab[1][0] + tab[0][1])
    f1 = f1_score(pred_flat, label_flat)

    class_score_flat = class_score.copy()
    class_score_flat = class_score_flat.flatten(order="F")
    MAPRC, auprc = auc(label_flat, class_score_flat)

    return acc, f1, auprc, MAPRC
