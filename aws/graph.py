import numpy as np
from bokeh.plotting import figure


def build_hist(df, column, label=None, w=450, h=300, bins=20, color="lightblue"):
    """
       build an histogram of the column col.
    """
    mu = df[column].mean()
    std = df[column].std()
    label = label if label else column
    title = u"%s (mu=%.2e, sigma=%.2e)" % (label, mu, std)
    f = figure(title=title, width=w, height=h, y_axis_type="log")
    hist, edges = np.histogram(df[column], density=True, bins=bins)
    f.quad(
        top=hist,
        bottom=0,
        left=edges[:-1],
        right=edges[1:],
        fill_color=color,
        line_color="grey")
    return f


def to_relative_time(df, device_to_endtime, rel_time_threshold=-100):
    """
    reindex data on a timescale that is relative to its failure
    """
    temp = df.reset_index()
    temp["failure_date"] = temp["device"].map(device_to_endtime)
    temp["dt_from_fail"] = (temp["date"] - temp["failure_date"]) / np.timedelta64(1, 'D')
    relative_time = temp.set_index(["device", "dt_from_fail"])

    # filter relative values
    relative_time = relative_time[relative_time.index.get_level_values("dt_from_fail") >= rel_time_threshold]
    relative_time = relative_time[relative_time.index.get_level_values("dt_from_fail") < 0]

    return relative_time
