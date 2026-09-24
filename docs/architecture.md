# Model architecture

TreeSegPy is a binary point classifier built on KPConv's deformable
Kernel Point Convolution, wrapped in a fully convolutional encoder–decoder.

## Input

Each point is represented by a **10-dimensional feature vector**:

| Index | Feature | Source |
|---|---|---|
| 0 | bias | constant 1.0 |
| 1 | z | patch-local height |
| 2 | linearity | local covariance eigenvalues |
| 3 | planarity | local covariance eigenvalues |
| 4 | sphericity | local covariance eigenvalues |
| 5 | omnivariance | local covariance eigenvalues |
| 6 | eigenentropy | local covariance eigenvalues |
| 7 | anisotropy | local covariance eigenvalues |
| 8 | verticality | smallest-eigenvalue eigenvector |
| 9 | density | distance to the 15th nearest neighbour |

The eight geometric features (indices 2–9) are computed in
`treesegpy/patch.py::compute_geometric_features` from the covariance of the
15 nearest neighbours of each point. Bias and z are added in
`treesegpy/batch.py::make_batch`.

## Feature definitions

For the three eigenvalues λ1 ≥ λ2 ≥ λ3 of the local covariance matrix:

- linearity = (λ1 − λ2) / λ1
- planarity = (λ2 − λ3) / λ1
- sphericity = λ3 / λ1
- omnivariance = (λ1 λ2 λ3)^(1/3)
- eigenentropy = −Σ pᵢ ln pᵢ, where pᵢ = λᵢ / (λ1 + λ2 + λ3)
- anisotropy = (λ1 − λ3) / λ1
- verticality = 1 − |n_z|, where n is the eigenvector of λ3
- density = distance to the 15th nearest neighbour

## Network

The architecture is defined in `treesegpy/config.py`:
simple
resnetb
resnetb_strided
resnetb
resnetb
resnetb_strided
resnetb
resnetb
resnetb_strided
resnetb
resnetb
resnetb_strided
resnetb
resnetb
nearest_upsample
unary
nearest_upsample
unary
nearest_upsample
unary
nearest_upsample
unary


Four strided ResNet blocks in the encoder, four nearest-neighbour upsampling
stages in the decoder, with skip connections between corresponding levels.

## KPConv configuration

- `first_features_dim = 64`
- `num_kernel_points = 15`
- `first_subsampling_dl = 0.05` m
- `max_neighbors_per_layer = [15, 15, 15, 15, 15]`
- Deformable convolution enabled: `deform_radius = 5.0`,
  `deform_fitting_mode = 'point2point'`
- Batch normalisation enabled

## Output

Per-point logits of shape (N, 2). Class 0 is non-tree, class 1 is tree. The
training loss is cross-entropy with balanced class weights `[0.5, 0.5]`.