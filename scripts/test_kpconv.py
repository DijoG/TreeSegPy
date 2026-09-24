import numpy as np
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors

pts = np.random.rand(10000, 3).astype(np.float32)
feats = np.random.rand(10000, 3).astype(np.float32)
labels = np.zeros(10000, dtype=np.int32)

sub_pts, sub_feats, sub_lab = cpp_subsampling.subsample(
    pts, features=feats, classes=labels,
    sampleDl=float(0.1), method='barycenters', verbose=0,
)
print('subsample OK:', sub_pts.shape, 'from', pts.shape)

queries = sub_pts[:100].astype(np.float32)
supports = sub_pts.astype(np.float32)
q_batches = np.array([0, len(queries)], dtype=np.int32)
s_batches = np.array([0, len(supports)], dtype=np.int32)

inds = cpp_neighbors.batch_query(
    queries, supports, q_batches, s_batches, radius=float(0.2),
)
print('batch_query OK:', len(inds), 'queries resolved')
