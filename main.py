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
from src.vizualization import plot_results


if __name__ == '__main__':
    model = Model()

    dataset_conf = read_yaml('configs/dataset.yaml')
    train_conf = read_yaml('configs/train.yaml')

    # train datasets
    dataset_files = scan_files(dataset_conf['data']['images'])
    train_length = round(len(dataset_files) * 0.8)
    train_dataloader = DataLoader(
        TrainDataset(
            data=dataset_files[:train_length],
            in_classes=dataset_conf['in_classes'],
            out_classes=dataset_conf['out_classes'],
            preprocess=train_augmentations(),
            dataset_path=dataset_conf['data']
        ),
    )
    # val dataset
    val_dataloader = DataLoader(
        ValDataset(
            data=dataset_files[train_length:],
            in_classes=dataset_conf['in_classes'],
            out_classes=dataset_conf['out_classes'],
            preprocess=val_augmentations(),
            dataset_path=dataset_conf['data']
        ),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=train_conf['optimizer']['lr'])
    loss_fn = Loss.get_instance(
        adapter=train_conf['loss']['adapter'],
        loss_fn=train_conf['loss']['loss_fn']
    )

    for epoch in range(5):
        # train
        stats = {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }
        model.train()
        itm = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for batch_idx, (image, mask) in itm:
            image = image.to(train_conf['device'])
            mask = mask.to(train_conf['device'])

            optimizer.zero_grad()
            out_mask = model(image)

            loss = loss_fn(out_mask, mask)
            loss.backward()
            optimizer.step()

            out_mask = out_mask.detach() > 0.5

            tp, fp, fn, tn = smp.metrics.get_stats(out_mask.long(), mask.long(), num_classes=3, mode="multiclass")
            stats['tp'] += tp
            stats['fp'] += fp
            stats['fn'] += fn
            stats['tn'] += tn
            # plot_results(image, mask, out_mask)

        print('train-scores', 'epoch', epoch)
        print('iou', smp.metrics.iou_score(**stats, reduction="micro-imagewise"))
        print('f1_score', smp.metrics.f1_score(**stats, reduction="micro-imagewise"))
        print('accuracy', smp.metrics.accuracy(**stats, reduction="micro-imagewise"))
        print('precision', smp.metrics.precision(**stats, reduction="micro-imagewise"))
        print('recall', smp.metrics.recall(**stats, reduction="micro-imagewise"))

        # val
        model.eval()
        with torch.no_grad():
            # main
            itm = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
            stats = {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "tn": 0,
            }

            for batch_idx, (image, mask) in itm:
                image = image.to(train_conf['device'])
                mask = mask.to(train_conf['device'])

                out_mask = model(image)
                out_mask = out_mask.detach() > 0.5

                tp, fp, fn, tn = smp.metrics.get_stats(out_mask.long(), mask.long(), num_classes=3, mode="multiclass")

                stats['tp'] += tp
                stats['fp'] += fp
                stats['fn'] += fn
                stats['tn'] += tn

            print('val-scores', 'epoch', epoch)
            print('iou', smp.metrics.iou_score(**stats, reduction="micro-imagewise"))
            print('f1_score', smp.metrics.f1_score(**stats, reduction="micro-imagewise"))
            print('accuracy', smp.metrics.accuracy(**stats, reduction="micro-imagewise"))
            print('precision', smp.metrics.precision(**stats, reduction="micro-imagewise"))
            print('recall', smp.metrics.recall(**stats, reduction="micro-imagewise"))

    torch.save(model.state_dict(), './models/semantic.pth')
