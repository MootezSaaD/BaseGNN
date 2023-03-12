import pandas as pd

def static_splitter(src):
    _df = pd.read_csv(src)
    col_name = 'split'
    splits = []

    for split in {'train', 'test', 'eval'}:
        splits.append(_filter_by_col(_df, col_name, split))

    return splits
    

def _filter_by_col(df, col, val):
    return df[df[col] == val]