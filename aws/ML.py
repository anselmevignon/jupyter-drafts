import numpy as np
import pandas as pd

SUSPICIOUS_POSITIVES = set(["S1F0GPFZ", "S1F136J0", "W1F0KCP2", "W1F0M35B", "W1F11ZG9"])


def filter_devices(df):
    "remove suspicious devices"
    return df.filter(axis="index", items=(set(df.index) - SUSPICIOUS_POSITIVES))


def build_deriv(df, c, n=1):
    "build degree n temporal derivative for an attribute"
    def per_device(per_device):
        clean_index = per_device.reset_index(level=1, drop=True)
        resampled = clean_index.resample('1D').pad()

        raw_diff = np.diff(resampled, n=n)
        # fill the series start with zeros
        while (len(raw_diff) < len(resampled)):
            raw_diff = np.insert(raw_diff, 0, 0)
        d = pd.Series(data=raw_diff, index=resampled.index)
        return d.dropna()[d > 0]
    some_d = df[c].groupby(level="device").apply(per_device)
    return some_d.swaplevel()


def resample_per_device(df):
    "make sure there are no missing values in the dset"
    if df.index.names == ["device", "date"]:
        df = df.swaplevel().sort_index()
    groups = df.groupby(level="device")
    sampled = (
        groups.get_group(g).reset_index(level="device").resample("1D").pad().reset_index()
        for g in groups.groups)
    return pd.concat(sampled).set_index(["date", "device"])


def subsample_negatives(frac, label_set, feature_set):
    "remove negative samples of the dataset"
    idx = pd.IndexSlice
    set_size = label_set.shape[0]
    pos_size = label_set[label_set["failure"] > 0].shape[0]
    print(" %d with %d positives" % (set_size, pos_size))
    sub_label = label_set[label_set["failure"] > 0].append(label_set[label_set["failure"] == 0].sample(frac=frac))
    sub_feature = feature_set.loc[idx[sub_label.index]]
    sub_size_0 = sub_feature.shape[0]
    print("new size %i" % sub_size_0)
    return sub_label, sub_feature
