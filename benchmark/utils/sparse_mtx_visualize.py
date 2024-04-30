# import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.io import mmread

files = [f for f in Path('/home/jtrzcinski/ba_thesis_ginkgo/ginkgo/build/develop_wo_mpi_hip/benchmark').iterdir() if f.is_file() and f.suffix == '.mtx']
print(files)
for file in files:
    sparse_matrix = csr_matrix(mmread(file))
    nonzero_indices = sparse_matrix.nonzero()
    nonzero_values = sparse_matrix.data
    plt.scatter(nonzero_indices[1], nonzero_indices[0], c=nonzero_values, s=0.1, cmap='viridis')    
    plt.savefig(file.stem + '.png',dpi=400)
    plt.clf()
    sparse_matrix = None
    nonzero_indices = None
    nonzero_values = None
