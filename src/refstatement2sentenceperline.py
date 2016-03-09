import argparse

def main():
    parser = argparse.ArgumentParser(description="""Convert conllu to conll format""")
    parser.add_argument('input', help="")
    args = parser.parse_args()

    for line in open(args.input).readlines():
        line = eval(line.strip().replace('"referenceStatement" : ',""))
        print(line[0])




if __name__ == "__main__":
    main()
