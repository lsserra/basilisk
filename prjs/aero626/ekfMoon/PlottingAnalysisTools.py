import numpy as np
import matplotlib.pyplot as plt

def plot_landmark_innovations(
    innovations_log,
    xLabel="Time [s]",
    title="Landmark Innovations, Body Frame Axis",
    unitString = "km",
    show_measurement_noise=True,
    measurementNoiseSigma=None,
    show_confidence=True,
    figsize=(8,5)
):
    """
    Plot landmark innovations with optional ±3σ confidence bounds.

    Args:
        innovations_log (list[LandMarkInnovation]): list of innovation objects
            with attributes:
                - t : timestamp (float)
                - innovation : (3,1) or (3,) numpy array
                - innovationCov : (3,3) numpy array
                - landmarkId : int
        xLabel (str): x-axis label
        title (str): plot title
        show_measurement_noise (bool): plot gray ±3σ noise lines if True
        measurementNoiseSigma (float|None): noise std deviation (km or m)
        show_confidence (bool): plot ±3σ confidence lines from innovationCov
        figsize (tuple): matplotlib figure size
    """

    # --- Parse logs ---
    innTime_array = np.array([entry.t for entry in innovations_log])
    inn_array = np.array([entry.innovation.flatten() for entry in innovations_log])  # shape (N, 3)
    innSigma3_array = np.array([
        3.0 * np.sqrt(np.diag(entry.innovationCov))
        for entry in innovations_log
    ])  # shape (N, 3)

    # --- Plot ---
    fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True)
    labels = [f"x {unitString}", f"y {unitString}", f"z {unitString}"]

    for i, ax in enumerate(axs):
        ax.scatter(innTime_array, inn_array[:, i], marker='x', color='k', label=f'Innovation {labels[i]}')

        # Optional measurement noise bounds
        if show_measurement_noise and measurementNoiseSigma is not None:
            ax.axhline(y=3 * measurementNoiseSigma, color='gray', linestyle='--', label='Measurement Noise ±3σ')
            ax.axhline(y=-3 * measurementNoiseSigma, color='gray', linestyle='--')

        # Optional ±3σ filter confidence bounds
        if show_confidence:
            ax.plot(innTime_array, innSigma3_array[:, i], '-r', label='Innovation ±3σ confidence')
            ax.plot(innTime_array, -innSigma3_array[:, i], '-r')

        ax.grid(True)
        ax.set_ylabel(f'{labels[i]}')
        if i == 0:
            ax.legend(loc='upper right')
        if i == len(axs)-1:
            ax.set_xlabel(xLabel)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
