import argparse
import pandas as pd
from nltk.tokenize import wordpunct_tokenize
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.preprocessing import Normalizer
from collections import Counter, defaultdict
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from nltk.corpus import stopwords
from scipy.spatial.distance import cosine
from sklearn.dummy import DummyClassifier



class StatementPair:
    @staticmethod
    def stoplist():
        sl = stopwords.words("english") + ", \" : ' . ; ! ?".split()
        return sl

    def __init__(self,row_index,ref_statement,target_statement,annotation):
        self.ref_statement = ref_statement #sentence is a list of forms
        self.target_statement = target_statement
        self.row_index = int(row_index)
        self.label = 1 if annotation == "SAME" else 0

    def _word_venn_diagram(self):
        commonwords = set(self.ref_statement).intersection(set(self.target_statement))
        onlyref = set(self.ref_statement).difference(set(self.target_statement))
        onlytarget =  set(self.target_statement).difference(set(self.ref_statement))
        return  commonwords,onlyref,onlytarget

    def _get_average_vector(self,wordset,embeddings):
        wordvecs = [embeddings["DEFAULT"]]
        for word in wordset:
            if word in embeddings:
                 wordvecs.append(embeddings[word])

        wordvecs = np.array(wordvecs)
        return wordvecs.sum(axis=0)

    def a_dicecoeff(self):
        D = {}
        commonwords, onlyref, onlytarget = self._word_venn_diagram()

        D["a_dicecoeff"] = len(commonwords) / len(self.ref_statement)
        D["a_onlyref"] = len(onlyref) / len(commonwords)
        D["a_onlyref_stop"] = len([x for x in onlyref if x in self.stoplist()])/len(onlyref) if len(onlyref) > 0 else 0
        return D

    def b_lengths(self):
        D = {}
        D["b_lenref"] = len(self.ref_statement)
        D["b_lentarget"] = len(self.target_statement)
        D["b_lendiff"] = len(self.ref_statement)- len(self.target_statement)
        return D

    def d_bow(self):
        D = {}

        commonwords, onlyref, onlytarget = self._word_venn_diagram()
        for b in onlyref:
            D["d_r_"+b]=1
        return D

    def c_embeds(self,embeddings):
        D = {}
        commonwords, onlyref, onlytarget = self._word_venn_diagram()

        common_average_vector = self._get_average_vector(commonwords,embeddings) #THIS DOES NOT HELP
        #for i,v in enumerate(common_average_vector):
        #    D["c_com_"+str(i)]=v
        ref_average_vector = self._get_average_vector(onlyref, embeddings)
        for i, v in enumerate(ref_average_vector):
            D["c_ref_" + str(i)] = v
        #tgt_average_vector = self._get_average_vector(onlytarget, embeddings)
        #for i, v in enumerate(tgt_average_vector):
        #    D["c_tgt_" + str(i)] = v
        #co = cosine(ref_average_vector,tgt_average_vector)
        #D["c_dif"] = 0 if np.isnan(co) else co
        return D

    def featurize(self,embeddings):
        D = {}
        D.update(self.a_dicecoeff())
        D.update(self.b_lengths())
        D.update(self.c_embeds(embeddings))
        #D.update(self.d_bow())

        return D


def collect_features(input,embeddings,vectorize=True,generateFeatures=True):

    labels = []
    featuredicts = []


    for line in open(input).readlines():
        row_index, ref_statement, target_statement, annotation = line.strip().split("\t")
        ref_statement  = wordpunct_tokenize(ref_statement.lower())
        target_statement = wordpunct_tokenize(target_statement.lower())

        sp = StatementPair(row_index,ref_statement,target_statement,annotation)
        if generateFeatures and int(row_index) > 0:
            featuredicts.append(sp.featurize(embeddings))
            labels.append(sp.label)
    if vectorize:
        vec = DictVectorizer()
        norm = Normalizer()
        features = vec.fit_transform(featuredicts)#.toarray()
        labels = np.array(labels)
        return features, labels, vec
    else:
        return featuredicts,labels, None



def crossval(features, labels, vec,printcoeffs=False):
    maxent = LogisticRegression(penalty='l2')
    dummyclass = DummyClassifier("most_frequent")
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
        dummyclass.fit(TrainX_i,Trainy_i)
        maxent.fit(TrainX_i,Trainy_i)

        ypred_i = maxent.predict(TestX_i)
        ydummypred_i = dummyclass.predict(TestX_i)
        #coeffs_i = list(maxent.coef_[0])
        #coeffcounter_i = Counter(vec.feature_names_)
        #for value,name in zip(coeffs_i,vec.feature_names_):
        #    coeffcounter_i[name] = value

        acc = accuracy_score(ypred_i, Testy_i)
        pre = precision_score(ypred_i, Testy_i)
        rec = recall_score(ypred_i, Testy_i)
        f1 = f1_score(ypred_i, Testy_i)
        # shared task uses f1 of *accuracy* and recall!

        scores["Accuracy"].append(acc)
        scores["F1"].append(f1)
        scores["Precision"].append(pre)
        scores["Recall"].append(rec)

        acc = accuracy_score(ydummypred_i, Testy_i)
        pre = precision_score(ydummypred_i, Testy_i)
        rec = recall_score(ydummypred_i, Testy_i)
        f1 = f1_score(ydummypred_i, Testy_i)

        scores["dummy-Accuracy"].append(acc)
        scores["dummy-F1"].append(f1)
        scores["dummy-Precision"].append(pre)
        scores["dummy-Recall"].append(rec)

        #posfeats = posfeats.intersection(set([key for (key,value) in coeffcounter.most_common()[:20]]))
        #negfeats = negfeats.intersection(set([key for (key,value) in coeffcounter.most_common()[-20:]]))

    #print("Pervasive positive: ", posfeats)
    #print("Pervasive negative: ",negfeats)

    #scores = cross_validation.cross_val_score(maxent, features, labels, cv=10)
    #print("--")
    for key in sorted(scores.keys()):
        currentmetric = np.array(scores[key])
        print("%s : %0.2f (+/- %0.2f)" % (key,currentmetric.mean(), currentmetric.std()))
    print("--")
    if printcoeffs:

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

def load_embeddings(embedpath):
    E = {}
    for line in open(embedpath).readlines():
        a=line.split()
        E[a[0]]=np.array([float(x) for x in a[1:]])
    E["DEFAULT"]=(np.array([0]*len(E[list(E.keys())[0]]))) # zero times the size of the first embedding
    return E


def main():
    parser = argparse.ArgumentParser(description="""Export AMT""")
    parser.add_argument('--input', default="../res/dga_amt_simplemajority.tsv")
    parser.add_argument('--embeddings', default="/Users/hmartine/data/glove.6B/glove.6B.50d.txt")
    args = parser.parse_args()

    E = load_embeddings(args.embeddings)

    features, labels, vec = collect_features(args.input,embeddings=E)
    crossval(features, labels, vec)



if __name__ == "__main__":
    main()
