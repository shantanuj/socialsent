from socialsent import seeds
from socialsent import lexicons
from socialsent.polarity_induction_methods import random_walk, label_propagate_probabilistic, dist, densify
from socialsent.evaluate_methods import binary_metrics
from socialsent.representations.representation_factory import create_representation
from collections import defaultdict

if __name__ == "__main__":
    seeds_map=defaultdict(list)
    labeled_words=[]
    f = open('./socialsent/labeled_words.txt')
    for l in f:
        w, label = l.strip().split('\t')
        seeds_map[int(label)].append(w)
        labeled_words.append(w)
    unlabeled_words=[]
    for l in open('./socialsent/unlabeled_words.txt'):
        unlabeled_words.append(l.strip())

    embeddings = create_representation("GIGA", "data/example_embeddings/gensim_model_20.model.txt",
        set(unlabeled_words).union(set(labeled_words)))
    eval_words = [word for word in embeddings.iw if word not in set(labeled_words)]

    # Using SentProp with 10 neighbors and beta=0.99
    #polarities = random_walk(embeddings, seeds_map, beta=0.7, nn=10,
    #        sym=True, arccos=False)
    #point_estimates = dict([(w,polarities[w].most_common()[0][0]) for w in polarities])
    #print "sleep_with", polarities["sleep_with"]
    #print "boner", polarities["boner"]
    #print "finger", polarities["finger"]
    #print "pills", polarities["pills"]

    #polarities = label_propagate_probabilistic(embeddings, seeds_map)
    #point_estimates = dict([(w,polarities[w].most_common()[0][0]) for w in polarities])
    #print "sleep_with", polarities["sleep_with"]
    #print "boner", polarities["boner"]
    #print "finger", polarities["finger"]
    #print "pills", polarities["pills"]
    #acc, auc, avg_per  = binary_metrics(point_estimates, lexicon, eval_words)
    #print "Accuracy with best threshold: {:0.2f}".format(acc)
    #print "ROC AUC: {:0.2f}".format(auc)
    #print "Average precision score: {:0.2f}".format(avg_per)

    #polarities = dist(embeddings, seeds_map)
    #point_estimates = dict([(w,polarities[w].most_common()[0][0]) for w in polarities])
    #print "sleep_with", polarities["sleep_with"]
    #print "boner", polarities["boner"]
    #print "finger", polarities["finger"]
    #print "pills", polarities["pills"]


    polarities = densify(embeddings, seeds_map)
    point_estimates = dict([(w,polarities[w].most_common()[0][0]) for w in polarities])
    print "sleep_with", polarities["sleep_with"]
    print "boner", polarities["boner"]
    print "finger", polarities["finger"]
    print "pills", polarities["pills"]
