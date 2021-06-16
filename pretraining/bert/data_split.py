import sys
from progressbar import progressbar as tqdm

MIN_DATA_SIZE = 760000

file_name = sys.argv[1]
file_path = "/".join(file_name.split("/")[:-1])
print(file_name, file_path)

with open(file_name) as f:
    datas = f.readline()
