from socialsent import seeds
from socialsent import lexicons
from socialsent.polarity_induction_methods import random_walk, label_propagate_probabilistic, dist, densify
from socialsent.evaluate_methods import binary_metrics
from socialsent.representations.representation_factory import create_representation
from collections import defaultdict
import pickle
import pandas as pd
import sys
import numpy as np

if __name__ == "__main__":
    labeled_words_file = sys.argv[1]
    unlabeled_words_file = sys.argv[2]
    embeddings_file = sys.argv[3]
    output_file_prefix=sys.argv[4]
    seeds_map=defaultdict(list)
    labeled_words=[]
    f = open(labeled_words_file)
    for l in f:
        w, label = l.strip().split('\t')
        seeds_map[int(label)].append(w)
        labeled_words.append(w)
    unlabeled_words=[]
    for l in open(unlabeled_words_file):
        unlabeled_words.append(l.strip())

    embeddings = create_representation("GIGA", embeddings_file, set(unlabeled_words).union(set(labeled_words)))
    eval_words = [word for word in embeddings.iw if word not in set(labeled_words)]

    # Using SentProp with 10 neighbors and beta=0.99
    polarities = random_walk(embeddings, seeds_map, beta=0.7, nn=10, sym=True, arccos=False)
    point_estimates = dict([(w,polarities[w].most_common()[0][0]) for w in polarities if w in unlabeled_words])
    pickle.dump(polarities, open("{}_{}.pkl".format(output_file_prefix, "socialsent"),'wb'))
    df = pd.DataFrame().from_records(point_estimates.items(), columns=['word','label'])
    df.to_csv("{}_{}.csv".format(output_file_prefix, "socialsent"), sep='\t', encoding='utf-8')

    polarities = label_propagate_probabilistic(embeddings, seeds_map)
    point_estimates = dict([(w,polarities[w].most_common()[0][0]) for w in polarities if w in unlabeled_words])
    pickle.dump(polarities, open("{}_{}.pkl".format(output_file_prefix, "labelprop"),'wb'))
    df = pd.DataFrame().from_records(point_estimates.items(), columns=['word','label'])
    df.to_csv("{}_{}.csv".format(output_file_prefix, "labelprop"), sep='\t', encoding='utf-8')

    polarities = dist(embeddings, seeds_map)
    point_estimates = dict([(w,polarities[w].most_common()[0][0]) for w in polarities if w in unlabeled_words])
    pickle.dump(polarities, open("{}_{}.pkl".format(output_file_prefix, "dist"),'wb'))
    df.to_csv("{}_{}.csv".format(output_file_prefix, "dist"), sep='\t', encoding='utf-8')
