import matplotlib.pyplot as plt


def plot_results(image, gt_mask, pr_mask):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image.numpy().squeeze().transpose(1, 2, 0))  # convert CHW -> HWC
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt_mask.numpy().squeeze().transpose(1, 2, 0))  # just squeeze classes dim, because we have only one class
    plt.title("Ground truth")
    plt.axis("off")

    pr_mask = pr_mask.float()
    plt.subplot(1, 3, 3)
    plt.imshow(pr_mask.detach().numpy().squeeze().transpose(1, 2, 0))  # just squeeze classes dim, because we have only one class
    plt.title("Prediction")
    plt.axis("off")

    plt.show()
