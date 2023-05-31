import json
import sys
def parse():
    file = sys.argv[1]
    with open(file,"r") as f:
        json_trace = json.load(f)
    def myFunc(e):
        return e['ts']
    traceevents = json_trace['traceEvents']
    traceevents.sort(key=myFunc)
    json_trace['traceEvents']=traceevents
    with open("sample.json", "w") as outfile:
        json.dump(json_trace, outfile,indent=1)

if __name__ == "__main__":
    parse()
