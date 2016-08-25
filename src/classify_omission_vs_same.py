import argparse
import pandas as pd
from nltk.tokenize import wordpunct_tokenize
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from collections import Counter, defaultdict
from sklearn.feature_extraction import DictVectorizer
from nltk.corpus import wordnet as wn
import numpy as np




class StatementPair:
    def __init__(self,row_index,ref_statement,target_statement,annotation):
        self.ref_statement = ref_statement #sentence is a list of forms
        self.target_statement = target_statement
        self.row_index = int(row_index)
        self.label = 1 if annotation == "SAME" else 0

    def a_dicecoeff(self):
        D = {}
        D["a_dicecoeff"] = len(set(self.ref_statemen).intersection(set(self.target_statement))) / len(set(self.target_statement))
        return D

    def b_lengths(self):
        D = {}
        D["b_lenref"] = len(self.ref_statement)
        D["b_lentarget"] = len(self.target_statement)
        D["b_lendiff"] = len(self.ref_statement)- len(self.target_statement)

        return D

    def featurize(self):
        D = {}
        D.update(self.a_dicecoeff())
        D.update(self.b_lengths())
        return D


def collect_features(input,vectorize=True,generateFeatures=True):

    labels = []
    featuredicts = []


    for line in open(input).readlines():
        row_index, ref_statement, target_statement, annotation = line.strip().split("\t")
        ref_statement  = wordpunct_tokenize(ref_statement)
        target_statement = wordpunct_tokenize(target_statement)

        sp = StatementPair(row_index,ref_statement,target_statement,annotation)
        if generateFeatures:
            featuredicts.append(sp.featurize())
            labels.append(sp.label)

    if vectorize:
        vec = DictVectorizer()
        features = vec.fit_transform(featuredicts).toarray()
        labels = np.array(labels)
        return features, labels, vec
    else:
        return featuredicts,labels, None



def crossval(features, labels, vec):
    maxent = LogisticRegression(penalty='l1')
    #maxent = SGDClassifier(penalty='l1')
    #maxent = Perceptron(penalty='l1')
    maxent.fit(features,labels) # only needed for feature inspection, crossvalidation calls fit(), too
    coeffcounter = Counter(vec.feature_names_)
    negfeats = set(vec.feature_names_)
    posfeats = set(vec.feature_names_)

    scores = defaultdict(list)
    TotalCoeffCounter = Counter()

    for TrainIndices, TestIndices in cross_validation.KFold(n=features.shape[0], n_folds=10, shuffle=False, random_state=None):
        TrainX_i = features[TrainIndices]
        Trainy_i = labels[TrainIndices]

        TestX_i = features[TestIndices]
        Testy_i =  labels[TestIndices]

        maxent.fit(TrainX_i,Trainy_i)
        ypred_i = maxent.predict(TestX_i)
        coeffs_i = list(maxent.coef_[0])
        coeffcounter_i = Counter(vec.feature_names_)
        for value,name in zip(coeffs_i,vec.feature_names_):
            coeffcounter_i[name] = value

        acc = accuracy_score(ypred_i, Testy_i)
        pre = precision_score(ypred_i, Testy_i)
        rec = recall_score(ypred_i, Testy_i)
        # shared task uses f1 of *accuracy* and recall!
        f1 = 2 * acc * rec / (acc + rec)

        scores["Accuracy"].append(acc)
        scores["F1"].append(f1)
        scores["Precision"].append(pre)
        scores["Recall"].append(rec)

        posfeats = posfeats.intersection(set([key for (key,value) in coeffcounter.most_common()[:20]]))
        negfeats = negfeats.intersection(set([key for (key,value) in coeffcounter.most_common()[-20:]]))

    print("Pervasive positive: ", posfeats)
    print("Pervasive negative: ",negfeats)

    #scores = cross_validation.cross_val_score(maxent, features, labels, cv=10)
    print("--")

    for key in sorted(scores.keys()):
        currentmetric = np.array(scores[key])
        print("%s : %0.2f (+/- %0.2f)" % (key,currentmetric.mean(), currentmetric.std()))
    print("--")

    maxent.fit(features,labels) # fit on everything

    coeffs_total = list(maxent.coef_[0])
    for value,name in zip(coeffs_total,vec.feature_names_):
            TotalCoeffCounter[name] = value

    for (key,value) in TotalCoeffCounter.most_common()[:20]:
        print(key,value)
    print("---")
    for (key,value) in TotalCoeffCounter.most_common()[-20:]:
        print(key,value)
    print("lowest coeff:",coeffcounter.most_common()[-1])
    print("highest coeff",coeffcounter.most_common()[0])

def main():
    parser = argparse.ArgumentParser(description="""Export AMT""")
    parser.add_argument('--input', default="../res/dga_amt_simplemajority.tsv")
    args = parser.parse_args()

    for line in open(args.input).readlines():
        row_index, ref_statement, target_statement, annotation = line.strip().split("\t")
        ref_statement  = wordpunct_tokenize(ref_statement)
        target_statement = wordpunct_tokenize(target_statement)

    features, labels, vec = collect_features(args.input)
    crossval(features, labels, vec)



if __name__ == "__main__":
    main()
