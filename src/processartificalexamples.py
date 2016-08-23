import argparse




def normalized_text(s):
    s = s.lower()
    s = "".join([x for x in s if x.isalpha()])
    return s

def main():
    parser = argparse.ArgumentParser(description="""Convert conllu to conll format""")
    parser.add_argument('--input', default="/Users/hmartine/Dropbox/VerdiProjectFolder/binary_classifier_data_and_report/artificial_examples.txt")
    args = parser.parse_args()

    c = 0
    for line in open(args.input).readlines():
        line = line.replace("\t\t","\t") # silly patch, there are some duplicate tabs
        ref_statement, target_statement = line.strip().split("\t")
        c-=1
        lineparts = [str(c)]+line.strip().split("\t")
        out = [str(c),"ref_statement","ref_title","ref_url","target_statement","target_title","target_url","media_source","relation_type"]
        out[1] = ref_statement
        out[4] = target_statement

        print('\"'+'\",\"'.join(out)+'\"')



if __name__ == "__main__":
    main()
