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
from classify_omission_vs_same import StatementPair

trainingbow = set()

ner_path_bin = "/Users/hmartine/proj/verdisandbox/res/stanford-ner-2015-12-09/stanford-ner.jar"
ner_path_model = "/Users/hmartine/proj/verdisandbox/res/stanford-ner-2015-12-09/classifiers/english.conll.4class.distsim.crf.ser.gz"
ner_path_model = "/Users/hmartine/proj/verdisandbox/res/stanford-ner-2015-12-09/classifiers/english.muc.7class.distsim.crf.ser.gz"

from nltk.tag.stanford import StanfordNERTagger

def main():
    parser = argparse.ArgumentParser(description="""Export AMT""")
    parser.add_argument('--input', default="../res/dga_extendedamt_simplemajority.tsv")
    args = parser.parse_args()

    ner_tagger = StanfordNERTagger(ner_path_model,ner_path_bin)

    pairs = []

    for line in open(args.input).readlines():
        row_index, ref_statement, target_statement, annotation = line.strip().split("\t")
        ref_statement = wordpunct_tokenize(ref_statement)
        target_statement = wordpunct_tokenize(target_statement)

        sp = StatementPair(row_index, ref_statement, target_statement, annotation)
        pairs.append(sp)

    for sp in pairs[:10]:
        ref_ner = ner_tagger.tag(sp.ref_statement)
        print(ref_ner)



if __name__ == "__main__":
    main()
