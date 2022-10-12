import numpy as np


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