"""bonescan: dual-view bone scintigraphy segmentation (code release).

Modules:
    data      dataset, manifest reader, patient-level k-fold split
    augment   two-view augmentation, including paired soft copy-paste
    models    shared-encoder dual-view network with lesion and anatomy decoders
    losses    Dice + cross-entropy composite loss with deep supervision
    training  plain PyTorch training loop with early stopping
    mae       masked-autoencoder pre-training (Stage 1)
    plotting  training-curve figure helper
    addons    optional components referenced by losses/models (inactive in the paper configuration)
"""

__version__ = "0.1.0"
