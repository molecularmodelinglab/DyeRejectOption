import numpy as np
import random

def make_equal_sized_bins(y, num_cutoffs, num_sets=1):
    bottom_5 = np.quantile(y, 1 / 20)
    top_5 = np.quantile(y, 19 / 20)
    bin_size = np.abs(top_5 - bottom_5) / num_cutoffs

    if num_sets == 1:
        return [bottom_5 + (bin_size * x) for x in range(num_cutoffs)]
    else:
        bins = [[bottom_5 + (bin_size * x) for x in range(num_cutoffs)]]
        bin_step = bin_size / num_sets
        for _ in range(1, num_sets):
            bins.append([bottom_5 + (bin_size * x) + (bin_step * _) for x in range(num_cutoffs)])
        return bins


def ncs_descriptors(data, col_name, n_bits):
    unique_chemicals = list(set(data[col_name]))
    ncs_dict = {}
    for uniq_chem in unique_chemicals:
        assert uniq_chem not in ncs_dict.keys()
        generated_finger = [random.randint(0, 1) for _ in range(n_bits)]
        ncs_dict[uniq_chem] = generated_finger

    X1 = []
    for chem in data[col_name]:
        X1.append(ncs_dict[chem])

    return np.array(X1)
