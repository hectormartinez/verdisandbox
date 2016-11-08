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

trainingbow = set()

def read_dump(infile):
    names = "Ref TitleRef URLRef Target TitleTarget URLTarget Source Contains".split()
    frame = pd.read_csv(infile, sep="\t", names=names)
    return frame

def features_from_dump(infile,variant,embeddings,bowfilter):
    frame = read_dump(infile)
    refstatements = [wordpunct_tokenize(st) for st in list(frame.Ref)]
    targetstatements = [wordpunct_tokenize(st) for st in list(frame.Target)]
    featuredicts = []

    for i in range(len(refstatements)):
        sp = StatementPair(i, refstatements[i], targetstatements[i], 0)
        commonwords, onlyref, onlytarget = sp._word_venn_diagram()
        trainingbow.update(onlyref)
        featuredicts.append(sp.featurize(variant, embeddings,bowfilter))

    return featuredicts

def main():
    parser = argparse.ArgumentParser(description="""Export AMT""")
    parser.add_argument('--input', default="../res/dga_extendedamt_simplemajority.tsv")
    parser.add_argument('--dump_to_predict', default="../res/dga_data_october2016.tsv")
    parser.add_argument('--embeddings', default="/Users/hmartine/data/glove.6B/glove.6B.50d.txt")
    args = parser.parse_args()

    E = load_embeddings(args.embeddings)

    predarrays = {}

    variants = ["bcd","cd"]
    for variant in variants:
    #1 collect features for train
        trainfeatures, labels, vec = collect_features(args.input,embeddings=E,variant=variant,vectorize=False)
        maxent = LogisticRegression(penalty='l2')
        #TODO collect features for new data
        #TODO proper vectorization
        dumpfeatdicts = features_from_dump(args.dump_to_predict,variant=variant,embeddings=E,bowfilter=trainingbow)
        #dumpfeats = vec.fit_transform(dumpfeatdicts)
        vec = DictVectorizer()
        X_train = vec.fit_transform(trainfeatures)

        maxent.fit(X_train,labels)
        X_test = vec.transform(dumpfeatdicts)

        predarrays[variant+"_pred_label"] = ["SAME" if x == 0 else "OMISSION" for x in maxent.predict(X_test)]
        predarrays[variant + "_pred_prob"] = ['{:.2}'.format(y) for x,y in maxent.predict_proba(X_test)]


    #maxent.fit(np.array(allfeatures[:len(labels)]),labels)
    #print(maxent.predict(allfeatures[len(labels):]))
    # predict using {features, features without lenght} --> instance 'variants' properly
        #TODO compare prediction similarity
        #TODO provide an output format with labels and probs for both feature templates
    frame = read_dump(args.dump_to_predict)
    keyindices = sorted(predarrays.keys())

    header = "Index Ref TitleRef URLRef Target TitleTarget URLTarget Source Contains BCD_label BCD_prob CD_label CD_prob".replace(" ","\t")

    print(header)
    for a in zip([str(x) for x in range(len(frame.Ref))],list(frame.Ref),list(frame.Target),list(frame.TitleRef),list(frame.URLRef),list(frame.TitleTarget),list(frame.URLTarget),list(frame.Source),list(frame.Contains),predarrays[keyindices[0]],predarrays[keyindices[1]],predarrays[keyindices[2]],predarrays[keyindices[3]]):
        print("\t".join(a))



if __name__ == "__main__":
    main()
