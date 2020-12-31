import numpy as np
from similaritymeasures import (
    frechet_dist,
    area_between_two_curves,
    pcm,
    dtw,
)


def evaluate_traces(G, traces, metric):
    """
    Use metric to evaluate the accuracy of G in representing traces.
    Traces should be a 2D array of points (no headings)
    Available metrics:
        'frechet' - Frechet Distance
        'area' - Area between curves
        'pcm' - Partial Curve Mapping
        'dtw' - Dynamic Time Warping
    """

    min_score = float('inf')
    max_score = -float('inf')
    avg_score = 0
    count = 0

    for trace in traces:
        count += 1

        _, point_path = G.match_trace(trace)
        point_path = np.array(point_path, dtype="float64")
        point_path = point_path[:, :2]

        trace = np.array(trace, dtype="float64")
        trace = trace[:, :2]

        if metric == "frechet":
            score = frechet_dist(point_path, trace)
        elif metric == "area":
            score = area_between_two_curves(point_path, trace)
        elif metric == "pcm":
            score = pcm(point_path, trace)
        elif metric == "dtw":
            score, _ = dtw(point_path, trace)
        else:
            print("Invalid metric")
            return
        min_score = min(min_score, score)
        max_score = max(max_score, score)
        avg_score = ((avg_score*(count - 1)) + score) / count

    print(min_score, max_score, avg_score)


    
