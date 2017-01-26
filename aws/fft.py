import numpy as np
from scipy.fftpack import fft


def to_fft(df):
    """
    per device fft calculation"""
    try:
        resampled = df.resample("1D", level="date").mean().fillna(method='pad')
    except:
        # if we cannot resample, this (usually) means that we are using a rolling aggregation, outputing
        # an nd.array rather than a df. the good news is, in this case I shoudl already have resampled.
        resampled = df.copy()
    n = len(resampled)
    return np.abs(fft(resampled))[n//2:]


def __peaks(line):
    sorted_by_used = sorted(enumerate(line), key=lambda t: t[1], reverse=True)
    boundaries = set()
    peaks = []
    for i, value in sorted_by_used:
        if i not in boundaries:
            peaks.append((i, value))
        # in any case, i neighbors cannot be peaks now.
        boundaries.add(i+1)
        boundaries.add(i-1)
    return peaks


def fft_peak(df, p=0, index_no_value=True):
    """
    peak detection"""
    fft = to_fft(df)
    all_peaks = __peaks(fft.tolist())
    if (len(all_peaks) > p):
        return all_peaks[p][0 if index_no_value else 1]
    else:
        return 0
