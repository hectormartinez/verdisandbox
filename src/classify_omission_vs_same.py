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
import itertools


trainingbow = set()

ner_path_bin = "/Users/hmartine/proj/verdisandbox/res/stanford-ner-2015-12-09/stanford-ner.jar"
ner_path_model = "/Users/hmartine/proj/verdisandbox/res/stanford-ner-2015-12-09/classifiers/english.conll.4class.distsim.crf.ser.gz"

from nltk.tag.stanford import StanfordNERTagger

class StatementPair:
    @staticmethod
    def stoplist():
        sl = stopwords.words("english") + ", \" : ' . ; ! ?".split()
        return sl

    def __init__(self,row_index,ref_statement,target_statement,annotation):
        self.ref_statement = ref_statement #sentence is a list of forms
        self.target_statement = target_statement
        self.row_index = int(row_index)
        self.label = 0 if annotation == "SAME" else 1

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
        wv =  wordvecs.sum(axis=0) / (len(wordvecs))
        return wv

    def a_dicecoeff(self):
        D = {}
        try:
            commonwords, onlyref, onlytarget = self._word_venn_diagram()

            D["a_dicecoeff"] = len(commonwords) / len(self.ref_statement)
            D["a_onlyref"] = len(onlyref) / len(commonwords)
        except:
            D["a_dicecoeff"] = 0
            D["a_onlyref"] = 0
        return D

    def b_lengths(self):
        D = {}
        D["b_lenref"] = len(self.ref_statement)
        D["b_lentarget"] = len(self.target_statement)
        D["b_lendiff"] = len(self.ref_statement)- len(self.target_statement)
        return D

    def c_bow(self,bowfilter=None):
        D = {}
        commonwords, onlyref, onlytarget = self._word_venn_diagram()
        if bowfilter:
            for b in onlyref:
                if b in bowfilter:
                    D["c_r_"+b]=1
        else:
            for b in onlyref:
                D["c_r_" + b] = 1
        return D

    def d_embeds(self,embeddings):
        D = {}
        commonwords, onlyref, onlytarget = self._word_venn_diagram()

        common_average_vector = self._get_average_vector(commonwords,embeddings) #THIS DOES NOT HELP
        #for i,v in enumerate(common_average_vector):
        #    D["c_com_"+str(i)]=v
        ref_average_vector = self._get_average_vector(onlyref, embeddings)
        for i, v in enumerate(ref_average_vector):
            D["d_ref_" + str(i)] = v
        #tgt_average_vector = self._get_average_vector(onlytarget, embeddings)
        #for i, v in enumerate(tgt_average_vector):
        #    D["c_tgt_" + str(i)] = v
        #co = cosine(ref_average_vector,tgt_average_vector)
        #D["c_dif"] = 0 if np.isnan(co) else co
        return D

    def e_stop(self):
        D = {}
        commonwords, onlyref, onlytarget = self._word_venn_diagram()
        D["e_prop_stop"] = len([x for x in onlyref if x in self.stoplist()])/len(onlyref) if len(onlyref) > 0 else 0
        return D

    def _ner_sequences(self,taggedarray):
        acc = []
        sequences = set()
        for w, t in taggedarray:
            if t != "O":
                acc.append(w)
            elif acc:
                sequences.add(" ".join(acc))
                acc = []
        if acc:
            sequences.add(" ".join(acc))
            acc = []
        return sequences



    def f_ner(self,ner_tagger):
        D = {}

        ref_ner = ner_tagger.tag(self.ref_statement)
        target_ner = ner_tagger.tag(self.target_statement)
        ref_seqs=self._ner_sequences(ref_ner)
        target_seqs=self._ner_sequences(target_ner)
        print(ref_seqs.difference(target_seqs))
        D["f_ner"] = len(ref_seqs.difference(target_seqs))
        return D

    def f_ner2(self,ner_tagger):
        D = {}

        onlyref_ner = set(ner_tagger.tag(self.ref_statement)).difference(set(ner_tagger.tag(self.target_statement)))

        onlyref_ner.difference(set([x for x in onlyref_ner if x in self.stoplist()]))
        onlyref_ner = set([x for x in onlyref_ner if x[0].isupper()])
        #D["f_nerproxy"] = len(onlyref_caps)


        acc = []
        sequences = set()
        for w in self.ref_statement:
            if w in onlyref_caps:
                acc.append(w)
            elif acc:
                sequences.add(" ".join(acc))
                acc = []

        if acc:
            sequences.add(acc)
            acc = []
        D["f_nerproxy2"] = len(sequences)



        return D




    def featurize(self,variant,embeddings,ner_tagger,bowfilter=None):
        D = {}
        if "a" in variant:
            D.update(self.a_dicecoeff())
        if "b" in variant:
            D.update(self.b_lengths())
        if "c" in variant:
            D.update(self.c_bow(bowfilter))
        if "d" in variant:
            D.update(self.d_embeds(embeddings))
        if "e" in variant:
            D.update(self.e_stop())
        if "f" in variant:
            D.update(self.f_ner(ner_tagger))
        return D


def collect_features(input,variant,embeddings,ner_tagger,vectorize=True,generateFeatures=True):

    labels = []
    featuredicts = []


    for line in open(input).readlines():
        row_index, ref_statement, target_statement, annotation = line.strip().split("\t")
        ref_statement  = wordpunct_tokenize(ref_statement)
        target_statement = wordpunct_tokenize(target_statement)

        sp = StatementPair(row_index,ref_statement,target_statement,annotation)
        if generateFeatures and int(row_index) > 0:
            featuredicts.append(sp.featurize(variant,embeddings,ner_tagger))
            labels.append(sp.label)
    if vectorize:
        vec = DictVectorizer()
        norm = Normalizer()
        features = vec.fit_transform(featuredicts)#.toarray()
        labels = np.array(labels)
        return features, labels, vec
    else:
        return featuredicts,labels, None



def crossval(features, labels,variant,printcoeffs=False):
    maxent = LogisticRegression(penalty='l2')
    dummyclass = DummyClassifier("most_frequent")
    #maxent = SGDClassifier(penalty='l1')
    #maxent = Perceptron(penalty='l1')
    maxent.fit(features,labels) # only needed for feature inspection, crossvalidation calls fit(), too


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
        #pre = precision_score(ypred_i, Testy_i,pos_label=1)
        #rec = recall_score(ypred_i, Testy_i,pos_label=1)
        f1 = f1_score(ypred_i, Testy_i,pos_label=1)

        scores["Accuracy"].append(acc)
        scores["F1"].append(f1)
        #scores["Precision"].append(pre)
        #scores["Recall"].append(rec)

        #
        # acc = accuracy_score(ydummypred_i, Testy_i)
        # pre = precision_score(ydummypred_i, Testy_i,pos_label=1)
        # rec = recall_score(ydummypred_i, Testy_i,pos_label=1)
        # f1 = f1_score(ydummypred_i, Testy_i,pos_label=1)
        #
        # scores["dummy-Accuracy"].append(acc)
        # scores["dummy-F1"].append(f1)
        # scores["dummy-Precision"].append(pre)
        # scores["dummy-Recall"].append(rec)

        #posfeats = posfeats.intersection(set([key for (key,value) in coeffcounter.most_common()[:20]]))
        #negfeats = negfeats.intersection(set([key for (key,value) in coeffcounter.most_common()[-20:]]))

    #print("Pervasive positive: ", posfeats)
    #print("Pervasive negative: ",negfeats)

    #scores = cross_validation.cross_val_score(maxent, features, labels, cv=10)
    #print("--")
    #for key in sorted(scores.keys()):
    #    currentmetric = np.array(scores[key])
        #print("%s : %0.2f (+/- %0.2f)" % (key,currentmetric.mean(), currentmetric.std()))
        #print("%s : %0.2f" % (key,currentmetric.mean()))
    print("%s %.2f (%.2f)" % (variant,np.array(scores["Accuracy"]).mean(),np.array(scores["F1"]).mean()))
    if printcoeffs:

        maxent.fit(features,labels) # fit on everything

        coeffs_total = list(maxent.coef_[0])
        for (key,value) in TotalCoeffCounter.most_common()[:20]:
            print(key,value)
        print("---")
        for (key,value) in TotalCoeffCounter.most_common()[-20:]:
            print(key,value)

def load_embeddings(embedpath):
    E = {}
    for line in open(embedpath).readlines():
        a=line.split()
        E[a[0]]=np.array([float(x) for x in a[1:]])
    E["DEFAULT"]=(np.array([0]*len(E[list(E.keys())[0]]))) # zero times the size of the first embedding
    return E


def main():
    parser = argparse.ArgumentParser(description="""Export AMT""")
    parser.add_argument('--input', default="../res/dga_extendedamt_simplemajority.tsv")
    parser.add_argument('--embeddings', default="/Users/hmartine/data/glove.6B/glove.6B.50d.txt")
    args = parser.parse_args()

    E = load_embeddings(args.embeddings)
    ner_tagger = StanfordNERTagger(ner_path_model,ner_path_bin)
    letter_ids = "abcdef"

    variants = []
    for k in range(1,6):
        variants.extend(["".join(x) for x in itertools.combinations(letter_ids,k)])
    print(variants)


    for variant in variants:# ["a","b","c","d","e","f","ab","ac","ad","ae","af","bc","bd","be","bf","cd","ce","cf","abc","cde","abd","abf","bcf","cef","abde","abdf","abcde","abcdef"]:

        features, labels, vec = collect_features(args.input,variant=variant,embeddings=E,ner_tagger=ner_tagger)
        crossval(features, labels,variant)



if __name__ == "__main__":
    main()
