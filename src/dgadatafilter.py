import argparse




def normalized_text(s):
    s = s.lower()
    s = "".join([x for x in s if x.isalpha()])
    return s

def main():
    parser = argparse.ArgumentParser(description="""Convert conllu to conll format""")
    parser.add_argument('--input', default="/Users/hmartine/Dropbox/VerdiProjectFolder/binary_classifier_data_and_report/dga-data_noreps.tsv")
    args = parser.parse_args()
    c = 0
    header = '"row_index","ref_statement","ref_title","ref_url","target_statement","target_title","target_url","media_source","relation_type"'


    print(header)
    for line in open(args.input).readlines():
        line = line.replace("\t\t","\t") # silly patch, there are some duplicate tabs
        ref_statement, ref_title, ref_url, target_statement, target_title, target_url, media_source, relation_type = line.strip().split("\t")
        if normalized_text(ref_statement) == normalized_text(target_statement):
            pass
        if c < 500:
            c+=1
            line = line.replace(",","&#44;").replace('"','&#34;')
            ref_statement, ref_title, ref_url, target_statement, target_title, target_url, media_source, relation_type = line.strip().split("\t")
            lineparts = [str(c)]+line.strip().split("\t")
            #out = [str(c),ref_statement,target_statement]
            print('\"'+'\",\"'.join(lineparts)+'\"')



if __name__ == "__main__":
    main()
