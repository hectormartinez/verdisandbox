from nltk.corpus import framenet as fn

from pandas import pandas as pd


def isFirstBeginner(framename):
    #A first beginner is a frame X such as the superFrameName of all its Inheritance relations is X (because it is alway the parent of the Inheritance)
    f = fn.frame_by_name(framename)
    for rel in f['frameRelations']:
        if (rel['type']['name']) == 'Inheritance':
            #if rel['superFrameName'] != framename:
            #    return False
            print(framename,rel['superFrameName'],rel['subFrameName'])
    return True


for fx in fn.frames()[:10]:
    if isFirstBeginner(fx['name']):
        print(fx['name'])

D={}
D['WHO']=['People']
D['WHAT']=['Event','Eventive_affecting']
D['WHERE']=['Locale']
D['WHY']=['Event','Eventive_affecting']
D['HOW']=['']


frames = pd.read_csv("../data/res/frametargetlexicon.tsv",sep="\t")

framenames=set(frames.framename)


#Find FN first begginers and see which ones yield a perfect match
#Find the second level children of the and repeat

