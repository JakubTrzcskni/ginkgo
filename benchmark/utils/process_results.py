import json
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from pathlib import Path        

data_dir = "/home/jtrzcinski/ba_thesis_ginkgo/ginkgo/build/develop_wo_mpi_hip/benchmark/ba_results"
files = [f for f in Path(data_dir).iterdir() if
         f.is_file() and f.name.endswith(".json") and f.name.startswith("log")]
print(files)

experiment_name = []
precond_name = []
apply_times = []
for file in files:
  precond_stats = json.load(open(file))
  for k in precond_stats:
    matrix = "{size}-{stencil}".format(size=k["size"], stencil=k["stencil"])
    precond_list = k["preconditioner"]
    nnz_percentage = (k["nonzeros"] / k["rows"] ** 2) * 100
    for p, q in precond_list.items():
      apply_time = q['apply']['time']
      experiment_name.append(matrix)
      precond_name.append(p)
      apply_times.append(apply_time)
d = {"experiment":experiment_name, "preconditioner version":precond_name,"apply_time":apply_times}
df = pd.DataFrame(data=d)
#df.groupby(['experiment'])
filtered_df = df[df['experiment'] == '10000-5pt']
res = sb.barplot(x='preconditioner version', y = 'apply_time', data = df)