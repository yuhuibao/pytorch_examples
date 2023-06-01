import json
import sys
import os
import pandas
import pandas as pd
import numpy as np


# import numba


class Kernel:
    def __init__(self):
        self.name = ""
        self.id = None
        self.duration = None
        self.starttime = None
        self.endtime = None
        self.correlationid = None


class Operator:
    def __init__(self):
        self.name = ""
        self.id = None
        self.duration = None
        self.starttime = None
        self.endtime = None
        self.correlationid = []
        self.cudaevents = []
        self.inputdims = ""
        self.allocevents = []
        self.kernelids = []
        self.allocids = []


class Buffer:
    def __init__(self):
        self.id = None
        self.ts = None
        self.address = None
        self.size = None


def l_prod(in_list):
    res = 1
    for _ in in_list:
        res *= _
    return res


def load_json(json_file):
    if json_file is None:
        print("Error: No json file found.")
        return
    print("Analyzing json file: {}".format(json_file))
    with open(json_file, "r") as f:
        json_trace = json.load(f)


# 这个版本是针对GCP的，不同pytorch profiler 生成的trace json文件里变量名都不一样， kernel项里没有externalid，没法判断，只能再加correlation id
# 先把cpu 层对应的launch kernel找到，然后再取出对应的correlation id, 然后再比较kernel 层的对应的correlation id

# 新增打印 input output grid block 信息


def dataprocess():
    file = sys.argv[1]
    # print(file)
    # files.sort(key=lambda x: int(x[5:-5:])) #正确排序方式
    # files.sort(key = lambda x:int(str(x.split('.')[0]).replace('trace','')))
    # print(
    #     'modelid,layerid,cpueventname,cudatime,cudatimenooverlap, cpueventduration,cpueventsstarttime,cpueventendtime,cpucorrelationid,inputdims,'
    #     'inputproducts,inputsize,convkernelsize,bias,kernelduration,kernelstarttime,kernelendtime,correlationid,blocksperSM,'
    #     'warpsperSM,stream,grid,block,kernelname')
    # print(
    #     "operatorid,operatorname,infoeventids"
    # )
    fileid = 0

    with open(file, "r") as f:
        # with open('./data/traceresnet50/trace1.json', "r") as f:
        json_trace = json.load(f)
    cpuevents = []
    profilerstarttime = 0

    def myFunc(e):
        return e["ts"]

    def myFunc1(e):
        return e["starttime"]

    traceevents = json_trace["traceEvents"]
    traceevents.sort(key=myFunc)

    buffers = []

    for event in traceevents:
        if (
            (event.get("cat", "").lower() == "cpu_op")
            or (event.get("cat", "").lower() == "operator")
            and event.get("ph", "").lower() == "x"
        ):
            dur = event["dur"]
            ts = event["ts"]
            te = ts + dur
            popitem = []
            aoperator = Operator()
            aoperator.name = event["name"]
            aoperator.duration = dur
            aoperator.starttime = ts
            aoperator.endtime = te
            aoperator.inputdims = event["args"].get("Input Dims", "")
            # aoperator.inputdims = event['args']['Input dims']
            # aoperator.inputdims = str(event['args']['Input Dims']).replace(' ','').replace(',',';')

            cpuevents.append(aoperator)

            for cpueventsitem in cpuevents:
                if (
                    (te <= cpueventsitem.endtime and ts > cpueventsitem.starttime)
                    or (te < cpueventsitem.endtime and ts >= cpueventsitem.starttime)
                    or (
                        te == cpueventsitem.endtime
                        and ts == cpueventsitem.starttime
                        and aoperator.name != cpueventsitem.name
                    )
                ):  # 040223, ts > to ts >=
                    # if te == cpueventsitem.endtime and ts == cpueventsitem.starttime:
                    # print(aoperator.name !=cpueventsitem.name)
                    # print(aoperator.name)
                    # print(cpueventsitem.name)
                    popitem.append(aoperator)  # 检索出要删除的多余项
                    if cpueventsitem.inputdims == "":
                        cpueventsitem.inputdims = aoperator.inputdims
                elif te >= cpueventsitem.endtime and ts < cpueventsitem.starttime:
                    popitem.append(cpueventsitem)

            for item in popitem:
                if item in cpuevents:
                    cpuevents.remove(item)

        elif (
            (
                (event.get("cat", "").lower() == "cuda_runtime")
                or (event.get("cat", "").lower() == "runtime")
            )
            and event.get("ph", "").lower() == "x"
            and event.get("name", "").lower() == "cudalaunchkernel"
        ):
            dur = event["dur"]
            ts = event["ts"]
            te = ts + dur
            correlationid = event["args"]["correlation"]
            for cpueventsitem in cpuevents:
                if cpueventsitem.endtime > te and cpueventsitem.starttime < ts:
                    cpueventsitem.correlationid.append(correlationid)
        elif event.get("cat", "").lower() == "cpu_instant_event":
            ts = event["ts"]
            for cpueventsitem in cpuevents:
                if ts < cpueventsitem.endtime and ts > cpueventsitem.starttime:
                    cpueventsitem.allocevents.append(event)

        elif event.get("name", "") == "Iteration Start: PyTorch Profiler":
            profilerstarttime = event.get("ts")

    for event in json_trace["traceEvents"]:
        if (
            event.get("cat", "").lower() == "kernel"
            and event.get("ph", "").lower() == "x"
        ):
            correlationid = event["args"]["correlation"]
            dur = event["dur"]
            ts = event["ts"]
            te = ts + dur

            for cpueventsitem in cpuevents:
                if correlationid in cpueventsitem.correlationid:
                    akernel = Kernel()
                    akernel.name = event["name"]
                    akernel.duration = dur
                    akernel.starttime = ts
                    akernel.endtime = te
                    akernel.correlationid = correlationid
                    cpueventsitem.cudaevents.append(akernel)
    operatorid = 0
    kernelid = 0
    allocid = 0
    for cpueventsitem in cpuevents:
        # inputproducts = l_prod(cpueventsitem.inputdims[0])
        # if cpueventsitem.inputdims != "":
        #     inputsize = cpueventsitem.inputdims[0]
        # else:
        #     inputsize = ""
        # if cpueventsitem.name == "aten::conv2d":
        #     bias = cpueventsitem.inputdims[2]
        #     kernelsize = cpueventsitem.inputdims[1][2:]  # Kw x Kh
        #     # print(kernelsize)
        # else:
        #     bias = 0
        #     kernelsize = 0
        cudatime = 0
        mincuda = 0
        maxcuda = 0
        cudatimenooverlap = 0

        for cudaeventsitem in cpueventsitem.cudaevents:
            cpueventsitem.kernelids.append(kernelid)
            kernelid += 1
            if mincuda == 0:
                mincuda = cudaeventsitem.starttime - profilerstarttime
            elif mincuda > cudaeventsitem.starttime - profilerstarttime:
                mincuda = cudaeventsitem.starttime - profilerstarttime
            if maxcuda < cudaeventsitem.endtime - profilerstarttime:
                maxcuda = cudaeventsitem.endtime - profilerstarttime
            cudatimenooverlap += cudaeventsitem.endtime - cudaeventsitem.starttime
        cudatime = maxcuda - mincuda

        operatorid += 1
        infoevents = []

        for allocevent in cpueventsitem.allocevents:
            a = {
                "starttime": allocevent["ts"] - profilerstarttime,
                "address": hex(allocevent["args"]["Addr"]),
                "size": allocevent["args"]["Bytes"],
                "type": "allocate",
                "cudatime": "",
                "cudatimeoverlap": "",
                "eventid": allocid,
            }
            infoevents.append(a)
            cpueventsitem.allocids.append(allocid)
            allocid += 1
        #     print(
        #         file,
        #         ",",
        #         cpueventsitem.name,
        #         ",",
        #         hex(allocevent["args"]["Addr"]),
        #         ",",
        #         allocevent["args"]["Bytes"],
        #         ",",
        #         allocevent["args"]["Total Allocated"],
        #     )

        for i, cudaeventsitem in enumerate(cpueventsitem.cudaevents):
            k = {
                "starttime": cudaeventsitem.starttime - profilerstarttime,
                "cudatime": cudatime,
                "cudatimeoverlap": cudatimenooverlap,
                "type": "kernel-" + cudaeventsitem.name.replace(",", ";"),
                "address": "",
                "size": "",
                "eventid": cpueventsitem.kernelids[i],
            }
            infoevents.append(k)
        infoevents.sort(key=myFunc1)

        # print(
        #     operatorid,
        #     ",",
        #     cpueventsitem.name,
        #     # ",",
        #     # cpueventsitem.duration,
        #     # ",",
        #     # cpueventsitem.starttime - profilerstarttime,
        #     # ",",
        #     # cpueventsitem.endtime - profilerstarttime,
        #     ",",
        #     str(cpueventsitem.infoeventids).replace(",", ";"),
        # )
        for e in infoevents:
            print(e)


if __name__ == "__main__":
    dataprocess()
