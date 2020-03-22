def get_p(config, labels, ranks):
    num_samples = len(labels)
    p_to_value = {}
    for p in config['find_p']:
        correct = 0
        for label, rank in zip(labels, ranks):
            if rank[label] + 1 <= p:
                correct = correct + 1
        p_to_value[p] = correct / num_samples
    return p_to_value
