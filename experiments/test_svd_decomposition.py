# Test code for memory profiling (we are using fil profiler).
# Run with:
#    SCIPY_USE_PROPACK=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
#        OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
#        fil-profile run test_svd_decomposition.py
from numpy.random import default_rng
from scipy import sparse

rng = default_rng(0)

# Build large binary sparse matrix
m = sparse.random(1000, 100_000, density=0.01, random_state=rng, format="csr")
m.data[:] = 1.0

# Compute SVD with the three methods, using 100 components
for method in ('arpack', 'lobpcg', 'propack'):
    print(f"Running SVD method {method}")
    u, s, vt = sparse.linalg.svds(m, k=100, solver=method)

print("Done.")