import scipy as sp

weighted_graph = False
# Taken from https://github.com/daiquocnguyen/GNN-ReGVD/blob/b5e6657b003ada704b7c75e4bd237c1eee1d8138/code/modelGNN_updates.py#L199
def utc(tokens, word_embeddings, args):
    # print('using window size = ', window_size)

    y = []
    doc_len_list = []
    vocab_set = set()

    doc_words = tokens
    doc_len = len(doc_words)

    doc_vocab = list(set(doc_words))
    doc_nodes = len(doc_vocab)

    doc_len_list.append(doc_nodes)
    vocab_set.update(doc_vocab)

    doc_word_id_map = {}
    for j in range(doc_nodes):
        doc_word_id_map[doc_vocab[j]] = j

    # sliding windows
    windows = []
    if doc_len <= args.window_size:
        windows.append(doc_words)
    else:
        for j in range(doc_len - args.window_size + 1):
            window = doc_words[j: j + args.window_size]
            windows.append(window)

    word_pair_count = {}
    for window in windows:
        for p in range(1, len(window)):
            for q in range(0, p):
                word_p_id = window[p]
                word_q_id = window[q]
                if word_p_id == word_q_id:
                    continue
                word_pair_key = (word_p_id, word_q_id)
                # word co-occurrences as weights
                if word_pair_key in word_pair_count:
                    word_pair_count[word_pair_key] += 1.
                else:
                    word_pair_count[word_pair_key] = 1.
                # bi-direction
                word_pair_key = (word_q_id, word_p_id)
                if word_pair_key in word_pair_count:
                    word_pair_count[word_pair_key] += 1.
                else:
                    word_pair_count[word_pair_key] = 1.

    row = []
    col = []
    weight = []
    features = []

    for key in word_pair_count:
        p = key[0]
        q = key[1]
        row.append(doc_word_id_map[p])
        col.append(doc_word_id_map[q])
        weight.append(word_pair_count[key] if weighted_graph else 1.)
    adj = sp.sparse.csr_matrix((weight, (row, col)), shape=(doc_nodes, doc_nodes))


    for k, v in sorted(doc_word_id_map.items(), key=lambda x: x[1]):
        features.append(word_embeddings[k])


    return adj, features

# Taken from https://github.com/daiquocnguyen/GNN-ReGVD/blob/b5e6657b003ada704b7c75e4bd237c1eee1d8138/code/modelGNN_updates.py#L273
def ifc(tokens, word_embeddings, args):

    # print('using window size = ', window_size)

    doc_words = tokens
    doc_len = len(doc_words)

    row = []
    col = []
    weight = []
    features = []

    if doc_len > args.window_size:
        for j in range(doc_len - args.window_size + 1):
            for p in range(j + 1, j + args.window_size):
                for q in range(j, p):
                    row.append(p)
                    col.append(q)
                    weight.append(1.)
                    #
                    row.append(q)
                    col.append(p)
                    weight.append(1.)
    else:  # doc_len < window_size
        for p in range(1, doc_len):
            for q in range(0, p):
                row.append(p)
                col.append(q)
                weight.append(1.)
                #
                row.append(q)
                col.append(p)
                weight.append(1.)

    adj = sp.sparse.csr_matrix((weight, (row, col)), shape=(doc_len, doc_len))
    if weighted_graph == False:
        adj[adj > 1] = 1.

    #
    for word in doc_words:
        feature = word_embeddings[word]
        features.append(feature)


    return adj, features
  