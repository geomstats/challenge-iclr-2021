def pretty_metrics(metrics):
    metrics['ix'] = metrics.index
    for col in ['Unnamed: 0', 'config.dnn_architecture', 
                'modeldir', 'config.b_init']:
        metrics.pop(col)
    return metrics