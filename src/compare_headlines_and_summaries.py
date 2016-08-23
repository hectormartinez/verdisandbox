from collections import defaultdict, Counter
import json
from pathlib import Path
from fnsentence import FrameArgument, FrameEntry, FrameSentence
from nltk.corpus import stopwords
sw_en = stopwords.words('english')

C = Counter()



def compare_fn_sentences(title, summary):


    titleframes = set(title.framename_list())
    summaryframes = set(summary.framename_list())
    fn_coincidence =  round(len(titleframes.intersection(summaryframes)),3)
    fn_prop_over_title  = round(len(titleframes.intersection(summaryframes)) / len(titleframes),3)
    fn_coincide_names =  "_" if fn_coincidence == 0 else ",".join(sorted(titleframes.intersection(summaryframes)))

    title_text = set([x.lower() for x in title.textlist[1:] if x.lower() not in sw_en])
    summary_text = set([x.lower() for x in summary.textlist[1:] if x.lower() not in sw_en])

    lex_overlap = round(len(title_text.intersection(summary_text)) / len(title_text),3)

    C.update(titleframes.intersection(summaryframes))
    print("\t".join([str(x) for x in [lex_overlap,fn_coincidence,fn_prop_over_title,fn_coincide_names," ".join(title.textlist[1:])]]))


def plain_fn_sentences(path):
    sentences = []
    parse_cols = defaultdict(list)
    frames = {}
    sent_no = 1
    for fn_line in path.open():
        fn_line = fn_line.strip()
        if not fn_line:
            frame_sentence = FrameSentence(text=None,
                                           textlist=[None] + parse_cols['form'],
                                           postags=[None] + parse_cols['form'],
                                           heads=[-1] + [0] *len( parse_cols['form']),
                                           deprels=[None] + parse_cols['form'],
                                           preannotations=[None] + parse_cols['form'],
                                           frames=frames,
                                           id_="{}-{}".format("", sent_no))
            sentences.append(frame_sentence)
            parse_cols = defaultdict(list)
            sent_no += 1
            frames = {}
        else:
            # Extract FrameNet information
            fn_parts = fn_line.strip("\n").split("\t")
            parse_cols["form"].append(fn_parts[1])
            frame_name = fn_parts[3]
            argument_dict = json.loads(fn_parts[4])

            if frame_name:
                arguments = {k: FrameArgument(k, *map(int, v.split(":")))
                             for k, v in argument_dict.items()}
                token_id = int(fn_parts[0])##int(parse_cols['idx'][-1])
                frames[token_id] = FrameEntry(frame_name, token_id, arguments)
            if len(parse_cols['form']) > 0:
                frame_sentence = FrameSentence(text=None,
                                              textlist=[None] + parse_cols['form'],
                                              postags=[None] + parse_cols['form'],
                                              heads=[-1] + [0] *len( parse_cols['form']),
                                              deprels=[None] + parse_cols['form'],
                                              preannotations=[None] + parse_cols['form'],
                                              frames=frames,
                                              id_="{}-{}".format("", sent_no))
                #print(frame_sentence.frames)
        #sentences.append(frame_sentence)
    return sentences



titles = plain_fn_sentences(Path("../res/titles2000.txt.fn.pred"))
summaries = plain_fn_sentences(Path("../res/summaries2000_lowercase.txt.fn.pred"))

print(len(titles),len(summaries))

for t,s in zip(titles,summaries):
    #print(" ".join(t.textlist[1:]),"::::"," ".join(s.textlist[1:]))
    compare_fn_sentences(t,s)

print(C.most_common())