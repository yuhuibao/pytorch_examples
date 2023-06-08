import pandas as pd
import ast


def get_pagesAllocated(addr, size):
    if size < 0:
        size = 0 - size

    pages_to_return = []
    maxAddr = addr + size
    pageAddr = addr // 4096 * 4096
    while pageAddr <= maxAddr:
        pages_to_return.append(hex(pageAddr))
        pageAddr += 4096
    return pages_to_return


def convert_a2bid():
    listOfEvents = []
    input_file = "out2.txt"
    with open(input_file, "r") as f:
        for line in f:
            listOfEvents.append(ast.literal_eval(line))
    df = pd.DataFrame.from_dict(listOfEvents)
    # df["ReadAddr"]=""
    # df["WriteAddr"]=""
    df["AllocatedPages"] = ""

    nvbit_file = "../mem_trace_multilayer.txt"
    with open(nvbit_file, "r") as f:
        for line in f:
            if "Read" in line:
                data = line.split(":")[1]
                data = data.split(",")
                kernelid = int(data[1])
                readAddr = [e.strip() for e in data[2:] if e != "0x0"]
                # print(readAddr)
                df.loc[
                    (df["type"] != "allocate") & (df["eventid"] == kernelid),
                    ["ReadAddr"],
                ] = str(readAddr)
            if "Write" in line:
                data = line.split(":")[1]
                data = data.split(",")
                kernelid = int(data[1])
                writeAddr = [e.strip() for e in data[2:] if e != "0x0"]
                # print(writeAddr)
                df.loc[
                    (df["type"] != "allocate") & (df["eventid"] == kernelid),
                    ["WriteAddr"],
                ] = str(writeAddr)

    listOfEvents = []
    input_file_nvbit = "out2_nvbit.txt"
    with open(input_file_nvbit, "r") as f:
        for line in f:
            listOfEvents.append(ast.literal_eval(line))
    df_realaddr = pd.DataFrame.from_dict(listOfEvents)
    realaddr = df_realaddr["address"]
    for i, ra in enumerate(realaddr):
        df.loc[(df["type"] == "allocate") & (df["eventid"] == i), ["address"]] = ra

    df_realaddr["address"] = df_realaddr["address"].apply(lambda x: int(x, 0))
    df_realaddr["size"] = df_realaddr["size"].astype("int")
    df_realaddr[["address", "size"]]
    df_realaddr["col_3"] = df_realaddr.apply(
        lambda x: str(get_pagesAllocated(x["address"], x["size"])), axis=1
    )

    pageaddr = df_realaddr["col_3"]
    for i, pa in enumerate(pageaddr):
        df.loc[
            (df["type"] == "allocate") & (df["eventid"] == i), ["AllocatedPages"]
        ] = pa

    mapping = {}  # bufferid -> addr
    curid = 0
    addr_set = set()
    for index, row in df.iterrows():
        if row["type"] == "allocate":
            # print(index, ',',row['AllocatedPages'], row['size'])
            page_list = ast.literal_eval(row["AllocatedPages"])
            local_mapping = {}
            print("size = ", row["size"])
            for p in page_list:
                if p[:4] == "0x7f":
                    if p in addr_set:
                        if int(row["size"]) < 0:  # free
                            addr_set.remove(
                                p
                            )  # does not modify curid, curid always increase even if free
                        else:  # no free but allocate at the same address
                            mapping[curid] = p
                            local_mapping[curid] = p
                            curid += 1
                    else:
                        addr_set.add(p)
                        mapping[curid] = p
                        local_mapping[curid] = p
                        curid += 1

            df.loc[index, "AllocatedIDs"] = str(local_mapping)


if __name__ == "__main__":
    convert_a2bid()
