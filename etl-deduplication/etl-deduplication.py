def deduplicate(records, key_columns, strategy):
    """
    Deduplicate records by key columns using the given strategy.
    """
    # Write code here
    selected = {}          
    key_first_index = {}   

    def none_count(record):
        return sum(v is None for v in record.values())

    for idx, record in enumerate(records):
        key = tuple(record[col] for col in key_columns)

        if key not in selected:
            selected[key] = record
            key_first_index[key] = idx
        else:
            if strategy == "last":
                selected[key] = record

            elif strategy == "most_complete":
                curr = selected[key]
                if none_count(record) < none_count(curr):
                    selected[key] = record
                # tie → keep first (do nothing)

            # strategy == "first" → do nothing

    # Preserve order of first appearance of each key
    ordered_keys = sorted(key_first_index, key=lambda k: key_first_index[k])

    return [selected[k] for k in ordered_keys]