import argparse
from numpy.random import choice, seed
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from scipy.spatial.distance import cosine
from collections import Counter
from itertools import combinations
from scipy.stats import pearsonr

def load_embeddings_file(file_name, sep=" ",lower=False):
    """
    load embeddings file
    """
    emb={}
    for line in open(file_name).readlines()[1:]:
        fields = line.split(sep)
        vec = [float(x) for x in fields[1:]]
        word = fields[0]
        if lower:
            word = word.lower()
        emb[word] = vec
    return emb, len(emb[word])

class SentencePair:
    @staticmethod
    def stoplist():
        sl = stopwords.words("english") + ", \" : ' . ; ! ? ( ) [ ] { } - _ '' # ".split()
        return sl

    def __init__(self,indices,sA,sB,dicecoeff):
        self.idxA = indices[0]
        self.idxB = indices[1]
        self.sentA_orig = sA
        self.sentB_orig = sB
        self.dicecoeff = float(dicecoeff)
        self.sentA = wordpunct_tokenize(self.sentA_orig)
        self.sentB = wordpunct_tokenize(self.sentB_orig)


    def __str__(self):
        return "\t".join([self.sentA_orig,self.sentB_orig,str(self.dicecoeff)[:5]])
    def word_venn_diagram(self):

        commonwords = [x for x in  set(self.sentA).intersection(set(self.sentB)) if x not in SentencePair.stoplist()]
        onlyA = [x for x in set(self.sentA).difference(set(self.sentB)) if x not in SentencePair.stoplist() and len(x) > 2]
        onlyB =  [x for x in set(self.sentB).difference(set(self.sentA)) if x not in SentencePair.stoplist() and len(x) > 2]
        flag = "-"
        if commonwords == []:
            flag = "EMPTY"
        return  flag,commonwords,onlyA,onlyB

def get_instances(infile):
    pairs = []
    lines = open(infile).readlines()
    for idx in range(0,len(lines),5):

        indices = lines[idx].strip().replace("]","").replace("[","").split()
        sentenceA = lines[idx+1].strip()[2:]
        sentenceB = lines[idx + 2].strip()[2:]
        dicecoeff = lines[idx + 3].strip().split("=")[1]
        pairs.append(SentencePair(indices,sentenceA,sentenceB,dicecoeff))

    return pairs

def get_pairwise_from_arbitrary_tabsep(infile):
    pairs = []
    for line in open(infile).readlines():
        blocks = line.lower().strip().split("\t")
        pairs.extend(list(combinations(blocks,2)))
    sentencepairs = []
    for p in pairs:
        sentencepairs.append(SentencePair([0,1],p[0],p[1],0.5))
    return sentencepairs


def get_similar_words(A,B,E,threshold=0.95):
    pairs = []
    for w_a in sorted([a for a in A if a.lower() not in SentencePair.stoplist() and a in E.keys()]):
        for w_b in sorted([b for b in B if b.lower() not in SentencePair.stoplist() and b in E.keys()]):
            #sim = cosine(E[w_a],E[w_b])
            sim = pearsonr(E[w_a],E[w_b])[0]
            if sim >= threshold:
                pairs.append((w_a,w_b,sim))
    return pairs


def get_best_pairs(A,B,E,n_pairs=3):
    C = Counter()
    for w_a in sorted([a for a in A if a.lower() not in SentencePair.stoplist() and a in E.keys()]):
        for w_b in sorted([b for b in B if b.lower() not in SentencePair.stoplist() and b in E.keys()]):
            C[(w_a,w_b)]=pearsonr(E[w_a],E[w_b])[0]
    return C.items()


def main2():
    parser = argparse.ArgumentParser(description="""This one is for mass treatment of corpus dataa""")
    parser.add_argument('--input', default="../res/actors-dga-final_for_paraphrase.txt")
    #parser.add_argument('--embeds', default="../res/querydump.5w.csv.embeds")
    #parser.add_argument('--embeds', default="../res/glove.6B.50d.txt")
    #parser.add_argument('--embeds', default="/Users/hmartine/data/embeds/poly_a/en.polyglot.txt")
    parser.add_argument('--embeds', default="../res/querydump.csv.2w.embeds")
    parser.add_argument('--sample', default=20)

    seed(112)

    args = parser.parse_args()
    E,l = load_embeddings_file(args.embeds)

    PC = Counter()
    pairs = get_instances(args.input)
    if args.sample:
        for p in pairs:
            flag, common, A, B = p.word_venn_diagram()
            dice=str(p.dicecoeff)[:5]
            #print("\t".join([flag," ".join(common)," ".join(A)," ".join(B),dice]))
            #print("\t".join([".".join(A), "-".join(B), dice]))
            #pairs = get_similar_words(A,B,E,threshold=0.98)
            #if pairs:
            #    print(pairs)
            for pair, score in get_best_pairs(A,B,E):
                PC[pair]=score

    print("#",args.embeds)
    for x,y in PC.most_common(1000):
        print("\t".join(x)+"\t"+str(y))



def main():
    parser = argparse.ArgumentParser(description="""This one is for those few handcurated paraphrase examples""")
    parser.add_argument('--input', default="../res/paraphrase_examples_for_stats.tsv")
    #parser.add_argument('--embeds', default="../res/querydump.csv.2w.embeds")
    #parser.add_argument('--embeds', default="../res/querydump.5w.csv.embeds")
    #parser.add_argument('--embeds', default="/Users/hmartine/data/embeds/poly_a/en.polyglot.txt")
    parser.add_argument('--embeds', default="../res/glove.6B.50d.txt")


    args = parser.parse_args()
    E,l = load_embeddings_file(args.embeds)

    PC = Counter()
    pairs = get_pairwise_from_arbitrary_tabsep(args.input)
    for p in pairs:
        flag, common, A, B = p.word_venn_diagram()
        for a,b,s in get_similar_words(A,B,E,threshold=0.80):
            PC[tuple(sorted([a,b]))] = s

    for ((a,b),s) in PC.most_common():
        print(a,b,s)


if __name__ == "__main__":
    main()
