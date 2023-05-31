import torch
from argparse import ArgumentParser


def add_dataset_args(parser=None):
    parser = parser or ArgumentParser()
    group = parser.add_argument_group("Dataset")
    group.add_argument(
        "--keep_low_involvement_cores_for_evaluation",
        action="store_true",
        default=False,
        help="If true, keep the low involvement cores with less than 40 pct involvement for evaluation.",
    )
    group.add_argument(
        "--benign_undersampling_kfold",
        type=int,
        default=None,
        help="If set, undersample benign cores as a kfold split.",
    )
    group.add_argument(
        "--augmentations_mode",
        type=str,
        default="none",
        choices=["none", "tensor_augs", "ultrasound_augs", "both"],
        help="Augmentations mode.",
    )
    group.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold to use for training.",
    )
    group.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of folds.",
    )
    group.add_argument(
        "--add_ood_dataset",
        action="store_true",
        default=False,
        help="If true, add the OOD dataset to the training set.",
    )
    group.add_argument(
        "--normalization", type=str, choices=["global", "instance"], default="instance"
    )
    return parser


def _label_transform(label):
    return torch.tensor(label).long()


def create_datasets(args):
    from trusnet.data.exact.nct02079025.cohort_selection import (
        get_cores_for_patients,
        get_patient_splits,
        remove_benign_cores_from_positive_patients,
        remove_cores_below_threshold_involvement,
        undersample_benign,
        undersample_benign_as_kfold,
    )

    train_patients, val_patients, test_patients = get_patient_splits(
        fold=args.fold, n_folds=args.n_folds
    )

    train_cores = get_cores_for_patients(train_patients)
    val_cores = get_cores_for_patients(val_patients)
    test_cores = get_cores_for_patients(test_patients)

    train_cores = remove_cores_below_threshold_involvement(train_cores, 40)
    if not args.keep_low_involvement_cores_for_evaluation:
        val_cores = remove_cores_below_threshold_involvement(val_cores, 40)
        test_cores = remove_cores_below_threshold_involvement(test_cores, 40)

    train_cores = remove_benign_cores_from_positive_patients(train_cores)
    if args.benign_undersampling_kfold is None:
        train_cores = undersample_benign(train_cores)
    else:
        train_cores = undersample_benign_as_kfold(
            train_cores,
        )[args.benign_undersampling_kfold]

    from trusnet.data.exact.nct02079025.dataset.rf_datasets import (
        PatchesDataset,
        PatchViewConfig,
    )
    from trusnet.data.exact.transforms import (
        TransformV3,
        TensorImageAugmentation,
        UltrasoundArrayAugmentation,
        Normalize,
    )

    patch_view_cfg = PatchViewConfig(
        needle_region_only=True,
        prostate_region_only=False,
    )

    norm = Normalize(mode=args.normalization)

    eval_transform = TransformV3(norm=norm)
    if args.augmentations_mode == "none":
        train_transform = eval_transform
    elif args.augmentations_mode == "tensor_augs":
        train_transform = TransformV3(
            norm=norm,
            tensor_transform=TensorImageAugmentation(
                random_resized_crop=True,
                random_affine_rotation=10,
                random_affine_translation=[0.1, 0.1],
            ),
        )
    elif args.augmentations_mode == "ultrasound_augs":
        train_transform = TransformV3(
            norm=norm,
            us_augmentation=UltrasoundArrayAugmentation(
                random_phase_shift=True,
                random_phase_distort=True,
                random_envelope_distort=True,
            ),
        )
    elif args.augmentations_mode == "both":
        train_transform = TransformV3(
            norm=norm,
            tensor_transform=TensorImageAugmentation(
                random_resized_crop=True,
                random_affine_rotation=10,
                random_affine_translation=[0.1, 0.1],
            ),
            us_augmentation=UltrasoundArrayAugmentation(
                random_phase_shift=True,
                random_phase_distort=True,
                random_envelope_distort=True,
            ),
        )
    else:
        raise ValueError("Unknown augmentations_mode")

    train_dataset = PatchesDataset(
        core_specifier_list=train_cores,
        patch_view_config=patch_view_cfg,
        transform=train_transform,
        target_transform=_label_transform,
    )
    val_dataset = PatchesDataset(
        core_specifier_list=val_cores,
        patch_view_config=patch_view_cfg,
        transform=eval_transform,
        target_transform=_label_transform,
    )
    test_dataset = PatchesDataset(
        core_specifier_list=test_cores,
        patch_view_config=patch_view_cfg,
        transform=eval_transform,
        target_transform=_label_transform,
    )

    if not args.add_ood_dataset:
        return train_dataset, val_dataset, test_dataset

    # we also need to create a dataset to test OOD
    from trusnet.data.exact.nct02079025.server.segmentation import (
        list_available_prostate_segmentations,
    )

    ood_test_cores = list(
        set(test_cores) & set(list_available_prostate_segmentations())
    )
    ood_patch_view_cfg = PatchViewConfig(
        needle_region_only=False,
        prostate_region_only=False,
        patch_strides=(1, 1),
    )
    ood_test_dataset = PatchesDataset(
        core_specifier_list=ood_test_cores,
        patch_view_config=ood_patch_view_cfg,
        transform=eval_transform,
        target_transform=_label_transform,
    )

    return train_dataset, val_dataset, test_dataset, ood_test_dataset
