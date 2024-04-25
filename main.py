from src.model import Model
from torch.utils.data import DataLoader
from src.dataset_parser import scan_files
from src.config_reader import read_yaml
from src.datasets.train_dataset import TrainDataset
from src.datasets.val_dataset import ValDataset
from tqdm import tqdm
from src.loss import Loss
import segmentation_models_pytorch as smp
import torch
from src.augmentations.train_augmentation import train_augmentations
from src.augmentations.val_augmentation import val_augmentations


if __name__ == '__main__':
    model = Model()

    dataset_conf = read_yaml('configs/dataset.yaml')
    train_conf = read_yaml('configs/train.yaml')

    # train datasets
    main_train_files = scan_files(dataset_conf['main']['train']['data'])
    custom_train_files = scan_files(dataset_conf['custom']['train']['data'])

    train_dataloader = DataLoader(
        TrainDataset(
            main_train_files[:len(main_train_files) * dataset_conf['main']['train']['consider']] +
            custom_train_files[:len(custom_train_files) * dataset_conf['custom']['train']['consider']],
            in_classes=dataset_conf['in_classes'],
            out_classes=dataset_conf['out_classes'],
            preprocess=train_augmentations(),
            slices=train_conf['preprocess']['train']['slices']
        )
    )

    main_val_files = scan_files(dataset_conf['main']['val']['data'])
    val_main_dataloader = DataLoader(
        ValDataset(
            main_val_files[:len(main_val_files) * dataset_conf['main']['val']['consider']],
            in_classes=dataset_conf['in_classes'],
            out_classes=dataset_conf['out_classes'],
            preprocess=val_augmentations()
        )
    )

    custom_val_files = scan_files(dataset_conf['custom']['val']['data'])
    val_custom_dataloader = DataLoader(
        ValDataset(
            custom_val_files[:len(custom_val_files) * dataset_conf['custom']['val']['consider']],
            in_classes=dataset_conf['in_classes'],
            out_classes=dataset_conf['out_classes'],
            preprocess=val_augmentations()
        )
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=train_conf['optimizer']['lr'])
    loss_fn = Loss.get_instance(
        adapter=train_conf['loss']['adapter'],
        loss_fn=train_conf['loss']['loss_fn']
    )

    for epoch in range(40):
        # train
        model.train()
        itm = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch_idx, (image, mask) in itm:
            image = image.to(train_conf['device'])
            mask = mask.to(train_conf['device'])

            if torch.sum(mask):
                optimizer.zero_grad()
                out_mask = model(image)

                loss = loss_fn(out_mask, mask)
                loss.backward()
                optimizer.step()

                out_mask = out_mask.detach() > 0

                tp, fp, fn, tn = smp.metrics.get_stats(out_mask.long(), mask.long(), mode="binary")
                print(smp.metrics.iou_score(tp=tp, fp=fp, fn=fn, tn=tn))

        # val
        model.eval()
        with torch.no_grad():
            # main
            itm = tqdm(enumerate(val_main_dataloader), total=len(val_main_dataloader))
            stats = {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 0,
            }

            for batch_idx, (image_original, mask_original, dimensions) in itm:
                image_original = image_original.to(train_conf['device'])
                mask_original = mask_original.to(train_conf['device'])

                original_width = dimensions[1]
                original_height = dimensions[0]

                # x +- padding
                y_start, y_stop = 0, train_conf['preprocess']['val']['slices']['height'] + 128
                while y_stop < original_height:
                    x_start, x_stop = 0, train_conf['preprocess']['val']['slices']['width'] + 128
                    while x_stop < original_width:
                        image = image_original[:, :, y_start:y_stop, x_start:x_stop]
                        mask = mask_original[:, :, y_start:y_stop, x_start:x_stop]

                        if torch.sum(mask):
                            out_mask = model(image)
                            out_mask = out_mask.detach() > 0

                            tp, fp, fn, tn = smp.metrics.get_stats(out_mask.long(), mask.long(), mode="binary")

                            stats['tp'] += tp
                            stats['fp'] += fp
                            stats['fn'] += fn
                            stats['tn'] += tn

                        x_start += train_conf['preprocess']['val']['slices']['width']
                        x_stop += train_conf['preprocess']['val']['slices']['width']

                    y_start += train_conf['preprocess']['val']['slices']['height']
                    y_stop += train_conf['preprocess']['val']['slices']['height']
                    print(x_start, x_stop, y_start, y_stop)

                iou = smp.metrics.iou_score(
                    **stats,
                    reduction="micro-imagewise"
                )
                print('micro-imagewise', iou)

            iou = smp.metrics.iou_score(
                **stats,
                reduction="micro-imagewise"
            )
            print('main-micro-imagewise', iou)

            # custom
            itm = tqdm(enumerate(val_custom_dataloader), total=len(val_custom_dataloader))
            stats = {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 0,
            }

            for batch_idx, (image_original, mask_original, dimensions) in itm:
                image_original = image_original.to(train_conf['device'])
                mask_original = mask_original.to(train_conf['device'])

                original_width = dimensions[1]
                original_height = dimensions[0]

                # x +- padding
                y_start, y_stop = 0, train_conf['preprocess']['val']['slices']['height'] + 128
                while y_stop < original_height:
                    x_start, x_stop = 0, train_conf['preprocess']['val']['slices']['width'] + 128
                    while x_stop < original_width:
                        image = image_original[:, :, y_start:y_stop, x_start:x_stop]
                        mask = mask_original[:, :, y_start:y_stop, x_start:x_stop]

                        if torch.sum(mask):
                            out_mask = model(image)
                            out_mask = out_mask.detach() > 0

                            tp, fp, fn, tn = smp.metrics.get_stats(out_mask.long(), mask.long(), mode="binary")

                            stats['tp'] += tp
                            stats['fp'] += fp
                            stats['fn'] += fn
                            stats['tn'] += tn

                        x_start += train_conf['preprocess']['val']['slices']['width']
                        x_stop += train_conf['preprocess']['val']['slices']['width']

                    y_start += train_conf['preprocess']['val']['slices']['height']
                    y_stop += train_conf['preprocess']['val']['slices']['height']
                    print(x_start, x_stop, y_start, y_stop)

                iou = smp.metrics.iou_score(
                    **stats,
                    reduction="micro-imagewise"
                )
                print('micro-imagewise', iou)

            iou = smp.metrics.iou_score(
                **stats,
                reduction="micro-imagewise"
            )
            print('custom-micro-imagewise', iou)

