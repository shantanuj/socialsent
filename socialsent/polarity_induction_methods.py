from socialsent import util
import functools
import numpy as np
from socialsent import embedding_transformer
from scipy.sparse import csr_matrix
from multiprocessing import Pool
from sklearn.linear_model import LogisticRegression, Ridge
from collections import Counter
from socialsent.graph_construction import similarity_matrix, transition_matrix

def normalize_counter(x):
    total = sum(x.values(), 0.0)
    for key in x:
            x[key] /= float(total)
    return x

"""
A set of methods for inducing polarity lexicons using word embeddings and seed words.
"""

def dist(embeds, seeds_map, normalize=True, **kwargs):
    polarities = {}
    sim_mat = similarity_matrix(embeds, **kwargs)
    for i, w in enumerate(embeds.iw):
        found = False
        for seed_list in seeds_map.values():
            if w in seed_list:
                found=True
                break
        if not found:
            polarities[w]=Counter()
            for seed_key, seed_list in seeds_map.items():
                pol = np.mean([sim_mat[embeds.wi[seed], i] for seed in seed_list])
                polarities[w][seed_key]=pol
            if normalize:
                polarities[w]=normalize_counter(polarities[w])
    return polarities


def pmi(count_embeds, seeds_map, normalize=True, smooth=0.01, **kwargs):
    """
    Learns polarity scores using PMI with seed words.
    Adapted from Turney, P. and M. Littman. "Measuring Praise and Criticism: Inference of semantic orientation from assocition".
    ACM Trans. Inf. Sys., 2003. 21(4) 315-346.

    counts is explicit embedding containing raw co-occurrence counts
    """
    w_index = count_embeds.wi
    c_index = count_embeds.ci
    counts = count_embeds.m
    polarities = {}
    for w in count_embeds.iw:
        found = False
        for seed_list in seeds_map.values():
            if w in seed_list:
                found=True
                break
        if not found:
            polarities[w]=Counter()
            for seed_key, seed_list in seeds_map.items():
                pol = sum(np.log(counts[w_index[w], c_index[seed]] + smooth) - np.log(counts[w_index[seed],:].sum()) for seed in seed_list)
                polarities[w][seed_key] = pol
            if normalize:
                polarities[w]=normalize_counter(polarities[w])

    return polarities

def densify_helper(embeddings, positive_seeds, negative_seeds, 
        transform_method=embedding_transformer.apply_embedding_transformation, **kwargs):
    """
#    Learns polarity scores via orthogonally-regularized projection to one-dimension
#    Adapted from: http://arxiv.org/pdf/1602.07572.pdf
#    """
    p_seeds = {word:1.0 for word in positive_seeds}
    n_seeds = {word:1.0 for word in negative_seeds}
    new_embeddings = embeddings
    new_embeddings = embedding_transformer.apply_embedding_transformation(
            embeddings, p_seeds, n_seeds, n_dim=1,  **kwargs)
    polarities = {w:new_embeddings[w][0] for w in embeddings.iw}
    return polarities

def densify(embeddings, seeds_map, transform_method=embedding_transformer.apply_embedding_transformation, **kwargs):
    """
#    Learns polarity scores via orthogonally-regularized projection to one-dimension
#    Adapted from: http://arxiv.org/pdf/1602.07572.pdf
#    """
    seeds_items=seeds_map.items() 
    polarities={}
    for w in embeddings.iw:
        polarities[w]=Counter()

    for i, seeds_item in enumerate(seeds_items):
        p_seeds = {word:1.0 for word in seeds_item[1]}
        neg_seeds_items = seeds_items[:]
        neg_seeds_items.remove(seeds_item)
        n_seeds={}
        for neg_seed_key, negative_seeds in neg_seeds_items:
            for word in negative_seeds:
                n_seeds[word]=1.0
        new_embeddings = embeddings
        print "Category", seeds_item[0]
        print "POS Seeds", p_seeds.keys()[:10]
        print "Neg Seeds", n_seeds.keys()[:10]
        new_embeddings = embedding_transformer.apply_embedding_transformation(embeddings, p_seeds, n_seeds, n_dim=1,  **kwargs)
        for w in embeddings.iw:
            polarities[w][seeds_item[0]]=new_embeddings[w][0]
    return polarities


def random_walk(embeddings, seeds_map, beta=0.9, normalize=True, **kwargs):
    """
    Learns polarity scores via random walks with teleporation to seed sets.
    Main method used in paper. 
    """
    def run_random_walk(M, teleport, beta, **kwargs):
        def update_seeds(r):
            r += (1 - beta) * teleport / np.sum(teleport)
        return run_iterative(M * beta, np.ones(M.shape[1]) / M.shape[1], update_seeds, **kwargs)

    seeds_map_dict={}
    for seed_key, seed_list in seeds_map.items():
        seeds_map_dict[seed_key] = {word:1.0 for i,word in enumerate(seed_list)}
    words = embeddings.iw
    M = transition_matrix(embeddings, **kwargs)
    scores_dict={}
    for seed_key, seeds in seeds_map_dict.items():
        scores_dict[seed_key]=run_random_walk(M, weighted_teleport_set(words, seeds), beta, **kwargs)
    polarities={}
    for i, w in enumerate(words):
        polarities[w]=Counter()
        for seed_key in scores_dict:
            polarities[w][seed_key]=scores_dict[seed_key][i]
        if normalize:
            polarities[w]=normalize_counter(polarities[w])
    return polarities


def label_propagate_probabilistic(embeddings, seeds_map, normalize=True, **kwargs):
    """
    Learns polarity scores via standard label propagation from seed sets.
    One walk per label. Scores normalized to probabilities. 
    """
    words = embeddings.iw
    M = transition_matrix(embeddings, **kwargs)
    teleport_set_map={}
    for seed_key, seed_list in seeds_map.items():
        teleport_set_map[seed_key]=teleport_set(words, seed_list)
    def update_seeds(r):
        idm= np.eye(len(seeds_map))
        for seed_key, w_indices in teleport_set_map.items():
            r[w_indices] = idm[seed_key]
        r /= np.sum(r, axis=1)[:, np.newaxis]
    r = run_iterative(M, np.random.random((M.shape[0], len(seeds_map))), update_seeds, **kwargs)
    polarities={}
    for i, w in enumerate(words):
        polarities[w]=Counter()
        for seed_key in seeds_map:
            polarities[w][seed_key]=r[i][seed_key]
        if normalize:
           polarities[w]=normalize_counter(polarities[w])
    return polarities

### HELPER METHODS #####

def teleport_set(words, seeds):
    return [i for i, w in enumerate(words) if w in seeds]

def weighted_teleport_set(words, seed_weights):
    return np.array([seed_weights[word] if word in seed_weights else 0.0 for word in words])

def run_iterative(M, r, update_seeds, max_iter=10000, epsilon=1e-8, **kwargs):
    for i in range(max_iter):
        last_r = np.array(r)
        r = np.dot(M, r)
        update_seeds(r)
        if np.abs(r - last_r).sum() < epsilon:
            print "Epsilon converged"
            break
    return r

### META METHODS ###

def _bootstrap_func(embeddings, seeds_map, boot_size, score_method, seed, **kwargs):
    np.random.seed(seed)
    seeds_map_sample = {}
    for seed_key, seed_list in seeds_map.items():
        seeds_map_sample[seed_key]=np.random.choice(seed_list, boot_size)
    polarities = score_method(embeddings, seeds_map_sample, **kwargs)
    for w in polarities:
        for seed_key, seed_list in seeds_map.items():
            if w in seed_list:
                polarities.pop(w, None)
                break
    return polarities

def bootstrap(embeddings, seeds_map, num_boots=10, score_method=random_walk,
        boot_size=7, return_all=False, n_procs=15, **kwargs):
    pool = Pool(n_procs)
    map_func = functools.partial(_bootstrap_func, embeddings,seeds_map,
            boot_size, score_method, **kwargs)
    polarities_list = pool.map(map_func, range(num_boots))
    if return_all:
        return polarities_list
    else:
        polarities = {}
        for word in polarities_list[0]:
            polarities[word] = Counter()
            for seed_key in seeds_map:
                polarities[word][seed_key] = np.mean([polarities_list[i][word][seed_key] for i in range(num_boots)])
        return polarities
