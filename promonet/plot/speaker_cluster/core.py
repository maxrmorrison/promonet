#Plot a cluster for speaker similarity, from https://github.com/resemble-ai/Resemblyzer/blob/master/demo_utils.py

import matplotlib.pyplot as plt
from umap import UMAP
import numpy as np

_default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
_my_colors = np.array([
    [0, 127, 70],
    [255, 0, 0],
    [255, 217, 38],
    [0, 135, 255],
    [165, 0, 165],
    [255, 167, 255],
    [97, 142, 151],
    [0, 255, 255],
    [255, 96, 38],
    [142, 76, 0],
    [33, 0, 127],
    [0, 0, 0],
    [183, 183, 183],
    [76, 255, 0],
], dtype=np.float) / 255

def plot_speaker_clusters(gts, reconstructs, gt_speakers, reconstruct_speakers, ax=None, colors=None, markers=None, legend=True, 
                     title="", file=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        
    # Compute the 2D projections. You could also project to another number of dimensions (e.g. 
    # for a 3D plot) or use a different different dimensionality reduction like PCA or TSNE.
    reducer = UMAP(**kwargs)
    all_reduced = reducer.fit_transform(np.append(gts, reconstructs, axis=0))
    gt_reduced = all_reduced[:gts.shape[0]]
    reconstruct_reduced = all_reduced[gts.shape[0]:]
    
    # Draw the projections
    colors = colors or _my_colors
    for i, speaker in enumerate(np.unique(np.append(gt_speakers, reconstruct_speakers))):
        gt_projs = gt_reduced[np.array([gt_speaker == speaker for gt_speaker in gt_speakers])]
        reconstruct_projs = reconstruct_reduced[np.array([recon_speaker == speaker for recon_speaker in reconstruct_speakers])]
        marker = "o" if markers is None else markers[i]
        label = speaker if legend else None
        ax.scatter(*gt_projs.T, c=[colors[i]], marker=marker, label=label + " GT")
        ax.scatter(*reconstruct_projs.T, c=[colors[i] * 0.5], marker=marker, label=label + " reconstruct")

    if legend:
        ax.legend(title="Speakers", ncol=2)
    ax.set_title(title)
    #ax.set_xticks([])
    #ax.set_yticks([])
    ax.set_aspect("equal")

    if file:
        plt.savefig(file, bbox_inches='tight', pad_inches=0)
    
    return fig