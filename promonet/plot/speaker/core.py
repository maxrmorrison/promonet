import itertools

import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP


###############################################################################
# Constants
###############################################################################


COLORS = np.array([
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
], dtype=float) / 255


###############################################################################
# Plot speaker embeddings
###############################################################################


def from_embeddings(
        centers,
        embeddings,
        ax=None,
        markers=None,
        legend=True,
        title='',
        file=None):
    # Maybe create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Add title
    ax.set_title(title)

    # Format
    center_speakers, center_embeddings = zip(*centers.items())
    center_embeddings = np.array([item.numpy() for item in center_embeddings])
    speakers, embeddings = zip(*embeddings.items())
    speakers = itertools.chain.from_iterable([
        [index] * len(embed) for index, embed in zip(speakers, embeddings)])
    embeddings = np.array([
        item.numpy() for item in itertools.chain.from_iterable(
            embeddings)])

    # Compute 2D projections
    projections = UMAP().fit_transform(
        np.append(center_embeddings, embeddings, axis=0))
    center_projections = projections[:center_embeddings.shape[0]]
    projections = projections[center_embeddings.shape[0]:]

    # Iterate over speakers
    for i, speaker in enumerate(center_speakers):

        # Get projections
        center_projection = center_projections[i]
        speaker_projections = projections[
            np.array([index == speaker for index in speakers])]

        # Style
        marker = 'o' if markers is None else markers[i]
        label = speaker if legend else None

        # Plot
        ax.scatter(
            *center_projection.T,
            c=[COLORS[i]],
            marker=marker,
            label=label + ' GT')
        ax.scatter(
            *speaker_projections.T,
            c=[COLORS[i] * 0.5],
            marker=marker,
            label=label + ' reconstruct')

    # Add legend
    if legend:
        ax.legend(title='Speakers', ncol=2)

    # Equal aspect ratio
    ax.set_aspect('equal')

    # Save to disk
    if file:
        plt.savefig(file, bbox_inches='tight', pad_inches=0)

    return fig
