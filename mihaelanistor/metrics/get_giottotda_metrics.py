from gtda.diagrams import PairwiseDistance

def get_pairwise_distance_metrics():

    metrics = {
        "bottleneck": PairwiseDistance(metric = "bottleneck"),
        "wasserstein": PairwiseDistance(metric = "wasserstein", metric_params = {"p": 1}),
        "betti": PairwiseDistance(metric = "betti"),
        "landscape": PairwiseDistance(metric = "landscape"),
        "silhouette": PairwiseDistance(metric = "silhouette"),
        "heat": PairwiseDistance(metric = "heat"),
        "persistence_image": PairwiseDistance(metric = "persistence_image"),
    }

    return metrics
