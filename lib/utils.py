import glob

def getLatestModelNum(exp_path):
    lst = glob.glob(exp_path + '/*.jl')
    if len(lst) == 0:
        return -1
    else:
        return max([int(file_name.split("/")[-1].split("_")[0]) for file_name in lst])
