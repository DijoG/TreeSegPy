"""Configuration for TreeSegPy training with KPConv."""
from pathlib import Path


class Config:
    """
    Configuration object for KPConv, adapted for TLS tree/non-tree segmentation.

    Field names match what KPConv's KPFCNN model reads from the config.
    Values are chosen for TLS forest patches (~100k points, 40 m tall, 0.05 m resolution).
    """

    # ---- Dataset identity ----
    dataset = "TreeScan"
    dataset_task = "cloud_segmentation"
    num_classes = 2

    # ---- Input geometry ----
    in_points_dim = 3       # xyz
    in_features_dim = 10   # [bias, z, linearity, planarity, sphericity,
                           #  omnivariance, eigenentropy, anisotropy,
                           #  verticality, density]
    in_radius = 6.0         # patch radius matches patch.py

    # ---- Architecture ----
    # 4 resnetb_strided, 4 nearest_upsample + unary pairs
    architecture = [
        'simple',
        'resnetb',
        'resnetb_strided',
        'resnetb',
        'resnetb',
        'resnetb_strided',
        'resnetb',
        'resnetb',
        'resnetb_strided',
        'resnetb',
        'resnetb',
        'resnetb_strided',
        'resnetb',
        'resnetb',
        'nearest_upsample',
        'unary',
        'nearest_upsample',
        'unary',
        'nearest_upsample',
        'unary',
        'nearest_upsample',
        'unary',
    ]

    # ---- KPConv specifics ----
    first_features_dim = 64
    # Maximum neighbors per point at each pyramid level.
    max_neighbors_per_layer = [15, 15, 15, 15, 15]# Caps the C++ neighbor search output so KPConv memory stays bounded.
    num_kernel_points = 15
    first_subsampling_dl = 0.05    # 5 cm voxel grid — matches α = 0.05 m in TopTreeSegR
    conv_radius = 2.5              # in units of first_subsampling_dl → 12.5 cm convolution radius
    deform_radius = 5.0
    KP_extent = 1.0
    KP_influence = 'linear'
    aggregation_mode = 'sum'
    fixed_kernel_points = 'center'
    modulated = False

    # ---- Deformable convolution ----
    deform_fitting_mode = 'point2point'
    deform_fitting_power = 1.0
    deform_lr_factor = 0.1
    repulse_extent = 1.2

    # ---- Normalization ----
    use_batch_norm = True
    batch_norm_momentum = 0.99

    # ---- Loss ----
    # Inverse-frequency weighting: 50% non-tree / 50% tree
    class_w = [0.5, 0.5]

    # ---- Training ----
    max_epoch = 200
    learning_rate = 1e-2
    momentum = 0.98
    lr_decay = 0.97
    lr_decay_steps = 5
    grad_clip_norm = 100.0
    batch_num = 6                 # patches per batch
    epoch_steps = 200             # batches per epoch (train)
    validation_size = 0.2
    val_epoch_steps = 50

    # ---- Augmentation ----
    augment_rotation = 'vertical'
    augment_scale_min = 0.9
    augment_scale_max = 1.1
    augment_scale_anisotropic = False
    augment_symmetries = [False, False, False]
    augment_noise = 0.001
    augment_color = 0.8

    # ---- I/O ----
    treescan_path = str(Path.home() / "projects/TreeSegPy/data/patches_geom")
    checkpoint_path = str(Path.home() / "projects/TreeSegPy/logs")
    input_threads = 8
    num_workers = 0     # KPConv C++ extensions are not fork-safe
    pin_memory = True


def make_config():
    """Return a fresh Config instance."""
    return Config()