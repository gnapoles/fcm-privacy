"""Real-world quasi-nonlinear FCM of the Lissos River basin.

Source
------
C. Papadopoulos, T. Bakas, M. Tyrovolas, D. Latinopoulos, I. Kagalou,
M. Spiliotis and C. Stylios, "Participatory Modeling and Scenario Analysis
for Managing Mediterranean River Basins Using Quasi-nonlinear Fuzzy
Cognitive Maps", Environmental Processes, vol. 12, no. 2, p. 23, 2025.
DOI 10.1007/s40710-025-00765-3.  Open access under CC BY 4.0.


"""

import numpy as np

CONCEPTS = [
    "Soil and water conservation techniques",
    "Water quality",
    "Precipitation",
    "Wet-period conditions",
    "Soil water reserves",
    "Reservoir water availability",
    "Surface water and groundwater",
    "Soil degradation",
    "Employment creation",
    "Water demand",
    "Groundwater intensive exploitation",
    "Surface-water intensive exploitation",
    "Dry season",
    "Irrigated arable land",
    "Forest resources",
    "Raising pastures and goats",
    "Riverbed modifications",
    "Floods",
    "Industry and tourism",
    "Livelihood of population and settlements",
    "Dry farming crops",
    "Wildfire",
]

# Table 1: aggregated interconnection matrix (rows = cause, columns = effect).
INTERCONNECTION = np.array([
    [0, 4, 0, 0, 4, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 2, 0, 1, 5, 0, 0, 0, 0, 0, -3, 0, 0, 0, 0, -2, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 2, 0, 2, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
    [1, 0, 0, 0, -1, -2, 1, 0, 1, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, -2, 0, 0, 0, -1, -3, -5, 0, 3, 0, -1, 0, 0, 0, 0, 0, 0, 0, 4, -1, 0],
    [0, 0, 0, 0, 0, 0, -3, -6, 0, 1, 2, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, -2, 0, 0, -2, 0, 0, 0, -3, 0, 0, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, -3, 0, 3, 4, -3, -3, 0, 0, 0, 0, 2, 0, 0, 4, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, -1],
    [1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 2, 0, 0, 0, 0, 0, -3, 0, 0, 1, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 4, 2, 2, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1, -4, 0, 0, 3, -2, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, -1, 0, 0, 1, 0, 0, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 3, 0, 0, 2, 2, 1, 0, 0, -1],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, -3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 0, 3, 0, 0, -1, 0],
], dtype=float)

# Table 2, row "Initial states".
INITIAL_STATE = np.array([
    -0.400, 0.425, 0.396, 0.600, -0.091, 0.500, 0.400, 0.400, 0.300, 0.250,
    0.800, -0.500, 0.200, 0.200, 0.600, 0.000, 0.400, 0.400, -0.200, -0.200,
    0.500, 0.200,
])

# Published centralities (Table 5), used to validate the transcription.
PUBLISHED_OD = np.array([
    2.167, 0.333, 2.500, 0.500, 0.667, 1.334, 0.000, 0.334, 1.000, 1.500,
    3.334, 2.500, 1.499, 3.667, 1.167, 1.501, 1.833, 2.000, 1.334, 2.334,
    0.833, 1.500,
])
PUBLISHED_ID = np.array([
    0.500, 1.333, 0.000, 0.500, 1.334, 1.167, 3.333, 2.167, 1.333, 3.166,
    2.000, 1.500, 0.500, 2.334, 0.500, 0.833, 1.500, 4.499, 0.500, 3.333,
    1.001, 0.501,
])


def weight_matrix():
    """W = I / max(|I|), following Eq. (12) of the source article."""
    W = INTERCONNECTION / np.max(np.abs(INTERCONNECTION))
    np.fill_diagonal(W, 0.0)
    return W


def out_degree(W):
    return np.sum(np.abs(W), axis=1)


def in_degree(W):
    return np.sum(np.abs(W), axis=0)


def centrality(W):
    return out_degree(W) + in_degree(W)


def validate(verbose=True):
    """Check the transcription against the published centrality values."""
    W = weight_matrix()
    od_err = np.max(np.abs(out_degree(W) - PUBLISHED_OD))
    id_err = np.max(np.abs(in_degree(W) - PUBLISHED_ID))
    if verbose:
        print(f"max |out-degree - published| = {od_err:.4f}")
        print(f"max |in-degree  - published| = {id_err:.4f}")
        print(f"concepts = {W.shape[0]}, non-zero causal links = {int(np.sum(W != 0))}")
    assert od_err < 2e-3 and id_err < 2e-3, "transcription does not match the source"
    return True


def scenarios():
    """The three published scenarios, used as initial conditions."""
    s1 = INITIAL_STATE.copy()
    s2 = INITIAL_STATE.copy()
    for i in (2, 3, 12, 13):        # C3, C4, C13, C14 decreased by 10 %
        s2[i] *= 0.9
    s2[9] *= 1.1                    # C10 increased by 10 %
    s3 = INITIAL_STATE.copy()
    for i in (2, 3, 12, 13):        # the same changes at 20 %
        s3[i] *= 0.8
    s3[9] *= 1.2
    return np.vstack([s1, s2, s3])


if __name__ == "__main__":
    validate()
