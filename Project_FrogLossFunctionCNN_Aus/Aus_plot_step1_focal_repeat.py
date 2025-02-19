    # -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 09:05:52 2021

@author: Administrator
"""

import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    # ax.tick_params(top=True, bottom=False,
    #                labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    # ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

# def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

# def annotate_heatmap_two(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

########################################################

final_focal_array = []
final_focal_mat = []

nRepeat = 5
for idx in range(nRepeat):        
    repeat_name = 'repeat_' + str(idx)
    
    method_list = []
    acc_focal_list = []
    
    focal_folder = r'.\0429\1D_Aus_org_focal'
    focal_list = os.listdir(os.path.join(focal_folder,repeat_name))
    for focal_name in focal_list:
        
        # focal_path = os.path.join(focal_folder, repeat_name, focal_name, 
        #                           'result_all_percent_0.8_winsize_0.2_winover_0.8',
        #                           'accuracy_fscore_kappa.csv')

        focal_path = os.path.join(focal_folder, repeat_name, focal_name, 
                                  'result_all_percent_0.8_winsize_8820_winover_0.8',
                                  'accuracy_fscore_kappa.csv')

    
        acf_data = pd.read_csv(focal_path, header=None)
        acf_value = acf_data.values   
        
        acc_focal_list.append(acf_value[1])
        
        method_list.append(focal_name)
    
    acc_focal_array = np.vstack(acc_focal_list)
    acc_focal_mat = acc_focal_array.reshape(4, 3)
    
    final_focal_array.append(acc_focal_array)
    final_focal_mat.append(acc_focal_mat)

plot_focal_mat = np.mean(final_focal_mat, axis=0)
plot_focal_mat_std = np.std(final_focal_mat, axis=0)

print(plot_focal_mat)
print(plot_focal_mat_std)

########################################################

strings = np.asarray([['±', '±', '±'],
                       ['±', '±', '±'],
                        ['±', '±', '±'],
                        ['±', '±', '±']])

labels = (np.asarray(["{0:.3f} {1} {2:.3f}".format(value1, string, value2)
                      for value1, string, value2 in zip(plot_focal_mat.flatten(),
                                                      strings.flatten(),
                                                      plot_focal_mat_std.flatten())])
         ).reshape(4, 3)


alpha = ['0.1','0.25','0.5','0.75']
gamma = ['2','4','6']

data = plot_focal_mat

fig, ax = plt.subplots()
im = sns.heatmap(data, annot=labels, fmt="", cmap='YlGn', ax=ax, linewidths=3)
ax.set_xlabel('Gamma value')
ax.set_ylabel('Alpha value')

row_labels = alpha
col_labels = gamma
ax.set_xticklabels(col_labels)
ax.set_yticklabels(row_labels)



########################################################
# Creating annotated heatmaps
# alpha = ['0.1','0.25','0.5','0.75']
# gamma = ['2','4','6']

# fig, ax = plt.subplots()

# im, cbar = heatmap(plot_focal_mat, alpha, gamma, ax=ax,
#                     cmap="YlGn")
# # texts = annotate_heatmap(labels, valfmt="{x:.3f}")
# texts = annotate_heatmap(labels)

# ax.set_xlabel('Gamma value')
# ax.set_ylabel('Alpha value')
# fig.tight_layout()
# plt.show()

# plt.savefig('./plot_new/' + result_name)












