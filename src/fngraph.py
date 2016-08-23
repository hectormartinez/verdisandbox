"""Yields a graph for FN instead of the unwieldy labyrinth of nested dicts
We use a philosophy similar to conllreader and put stuff in the nodes, maybe as dicts or maybe as a class"""

from nltk.corpus import framenet as fn
fn.propagate_semtypes()


framekeys = set()
frametypes = set()

for fx in fn.frames():
    for k in fx.keys():
        framekeys.add(k)
    if fx['semTypes']:
        for t in fx['semTypes']:
            frametypes.add(t['name'])
#We could read straight from the
print(frametypes)


#for k in framekeys:
#    print(k,fn.frames()[0][k])
#    print(k,fn.frames()[1][k])
#    print(k,fn.frames()[2][k])

