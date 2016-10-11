import argparse
from nltk.tokenize import wordpunct_tokenize
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.preprocessing import Normalizer
from collections import Counter, defaultdict
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from nltk.corpus import stopwords
from sklearn.dummy import DummyClassifier
import pandas as pd
import itertools

from classify_omission_vs_same import load_embeddings, collect_features
from classify_omission_vs_same import StatementPair
from sklearn.linear_model.logistic import LogisticRegression



def features_from_dump(infile,variant,embeddings):
    names = "Ref TitleRef URLRef Target TitleTarget URLTarget Source Contains".split()
    frame = pd.read_csv(infile,sep="\t",names=names)

    refstatements = [wordpunct_tokenize(st) for st in list(frame.Ref)]
    targetstatements = [wordpunct_tokenize(st) for st in list(frame.Target)]
    featuredicts = []

    for i in range(len(refstatements)):
        sp = StatementPair(i, refstatements[i], targetstatements[i], 0)
        featuredicts.append(sp.featurize(variant, embeddings))

    return featuredicts

def main():
    parser = argparse.ArgumentParser(description="""Export AMT""")
    parser.add_argument('--input', default="../res/dga_extendedamt_simplemajority.tsv")
    parser.add_argument('--dump_to_predict', default="../res/dga_data_october2016.tsv")
    parser.add_argument('--embeddings', default="/Users/hmartine/data/glove.6B/glove.6B.50d.txt")
    args = parser.parse_args()

    E = load_embeddings(args.embeddings)

    letter_ids = "abcdef"

    variants = []
    for k in range(1,6):
        variants.extend(["".join(x) for x in itertools.combinations(letter_ids,k)])
    print(variants)


    variant="a"
    #1 collect features for train
    features, labels, vec = collect_features(args.input,embeddings=E,variant=variant)
    maxent = LogisticRegression(penalty='l2')
    maxent.fit(features,labels)
    preds = maxent.predict(features)
    #TODO collect features for new data
    #TODO proper vectorization
    dumpfeatdicts = features_from_dump(args.dump_to_predict,variant=variant,embeddings=E)
    dumpfeats = vec.fit_transform(dumpfeatdicts)
    #TODO predict using {features, features without lenght} --> instance 'variants' properly
    maxent.predict(dumpfeats)
    #TODO compare prediction similarity
    #TODO provide an output format with labels and probs for both feature templates

    print([x-y for x,y in zip(labels,preds)])


if __name__ == "__main__":
    main()
