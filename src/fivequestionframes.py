from nltk.corpus import framenet as fn
fn.propagate_semtypes()
from pandas import pandas as pd


def NumLexU(framename):
    f = fn.frame_by_name(framename)
    if f['lexUnit']:
        return(" ".join(f['lexUnit']))
    else:
        return "_"

def isFirstBeginner(framename):
    #A first beginner is a frame X such as the superFrameName of all its Inheritance relations is X (because it is alway the parent of the Inheritance)
    f = fn.frame_by_name(framename)
    for rel in f['frameRelations']:
        if (rel['type']['name']) in ['Inheritance','Subframe','Using']:
            if rel['superFrameName'] != framename:
                return False
            #print(framename,rel['superFrameName'],rel['subFrameName'])
    return True


BeginList = []


for fx in fn.frames():
    if isFirstBeginner(fx['name']):
        BeginList.append(fx['name']+'\t'+str(NumLexU(fx['name']))+'\t_')

#print(len(BeginList))
print('\n'.join(sorted(BeginList)))


D={}
D['WHO']=['People']
D['WHAT']=['Event','Eventive_affecting']
D['WHERE']=['Locale']
D['WHY']=['Event','Eventive_affecting']
D['HOW']=['']


frames = pd.read_csv("../res/frametargetlexicon.tsv",sep="\t")

framenames=set(frames.framename)


#Find FN first begginers and see which ones yield a perfect match
#Find the second level children of the and repeat

