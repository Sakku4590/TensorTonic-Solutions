def target_encoding(categories, targets):
    means = {}
    counts = {}

    for c, t in zip(categories, targets):
        means[c] = means.get(c, 0) + t
        counts[c] = counts.get(c, 0) + 1

    for c in means:
        means[c] /= counts[c]

    return [means[c] for c in categories]