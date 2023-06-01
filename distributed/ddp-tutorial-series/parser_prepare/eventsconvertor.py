import sys
import ast
import pandas as pd


def parse_event():
    listOfEvents = []
    input_file = sys.argv[1]
    with open(input_file, "r") as f:
        for line in f:
            listOfEvents.append(line)
    df = pd.DataFrame(listOfEvents)
    df.to_csv(input_file.split(".")[0] + ".csv")


def parse_nvbit_results():
    pass


if __name__ == "__main__":
    parse_event()
