# todo:
# 1. make test function modifiable for each parameter
# 2. make test return image and psnr
# 3. run methods in a loop
# 4. save comparison based on difference_mean()
import pandas as pd
import deepinv as dinv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from deepinv.models import DRUNet
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.optim.optimizers import optim_builder
from deepinv.training_utils import test
from torchvision import transforms
from deepinv.utils.parameters import get_ProxPnP_params, get_GSPnP_params, get_DPIR_params
from deepinv.utils.demo import load_dataset, load_degradation
import cv2
import numpy as np


def PSNR(img_path_a, img_path_b):
    img_a = cv2.imread(img_path_a)
    img_b = cv2.imread(img_path_b)
    img_a = img_a.astype(np.float32)
    img_b = img_b.astype(np.float32)
    diff = np.abs(img_a - img_b)
    mse = np.mean(diff ** 2)
    psnr = 20 * np.log10(255) - 10 * np.log10(mse)
    return psnr


def difference_mean(img_path_a, img_path_b):
    img_a = cv2.imread(img_path_a)
    img_b = cv2.imread(img_path_b)
    img_a = img_a.astype(np.float32)
    img_b = img_b.astype(np.float32)
    diff = np.abs(img_a - img_b)
    diff_mean = np.mean(diff)
    return diff_mean * 1e-2


def test_pnp(iter='ADMM', BASE_DIR=Path("."), seed=0, method="DIPR", dataset_name="set3c",
             parameters_dict={
    "lambda": 0.01,
    "g_param": 0.1,
    "stepsize": 0.1,
    "max_iter": 100,
    "alpha": 0.5,
}):
    ORIGINAL_DATA_DIR = BASE_DIR / "datasets"
    DATA_DIR = BASE_DIR / "measurements"
    RESULTS_DIR = BASE_DIR / "results"
    DEG_DIR = BASE_DIR / "degradations"
    # Set the global random seed from pytorch to ensure reproducibility of the example.
    torch.manual_seed(seed)
    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    print("Using device: {}".format(device))

    img_size = 256 if torch.cuda.is_available() else 32
    val_transform = transforms.Compose(
        [transforms.CenterCrop(img_size), transforms.ToTensor()]
    )

    # Generate a motion blur operator.
    kernel_index = 1  # which kernel to chose among the 8 motion kernels from 'Levin09.mat'
    kernel_torch = load_degradation(
        "Levin09.npy", DEG_DIR / "kernels", kernel_index=kernel_index
    )
    kernel_torch = kernel_torch.unsqueeze(0).unsqueeze(
        0
    )  # add batch and channel dimensions
    dataset = load_dataset(
        dataset_name, ORIGINAL_DATA_DIR, transform=val_transform)
    noise_level_img = 0.03  # Gaussian Noise standard deviation for the degradation
    n_channels = 3  # 3 for color images, 1 for gray-scale images
    p = dinv.physics.BlurFFT(
        img_size=(n_channels, img_size, img_size),
        filter=kernel_torch,
        device=device,
        noise_model=dinv.physics.GaussianNoise(sigma=noise_level_img),
    )

    # Use parallel dataloader if using a GPU to fasten training,
    # otherwise, as all computes are on CPU, use synchronous data loading.
    num_workers = 4 if torch.cuda.is_available() else 0

    n_images_max = 1  # Maximal number of images to restore from the input dataset
    # Generate a dataset in a HDF5 folder in "{dir}/dinv_dataset0.h5'" and load it.
    operation = "deblur"
    measurement_dir = DATA_DIR / dataset_name / operation
    dinv_dataset_path = dinv.datasets.generate_dataset(
        train_dataset=dataset,
        test_dataset=None,
        physics=p,
        device=device,
        save_dir=measurement_dir,
        train_datapoints=n_images_max,
        num_workers=num_workers,
    )

    batch_size = 1
    dataset = dinv.datasets.HDF5Dataset(path=dinv_dataset_path, train=True)
    data_fidelity = L2()
    prior = PnP(denoiser=DRUNet(
        pretrained="download", train=False, device=device))
    model = optim_builder(
        iteration=iter,
        prior=prior,
        data_fidelity=data_fidelity,
        early_stop=True,
        max_iter=parameters_dict['max_iter'],
        verbose=True,
        params_algo=parameters_dict,
    )
    save_folder = RESULTS_DIR / method / operation / dataset_name
    wandb_vis = False
    plot_metrics = False
    plot_images = True
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    test(
        model=model,
        test_dataloader=dataloader,
        physics=p,
        device=device,
        plot_images=plot_images,
        save_folder=save_folder,
        plot_metrics=plot_metrics,
        verbose=True,
        wandb_vis=wandb_vis,
        plot_only_first_batch=False,
    )
    return str(save_folder / "images/GT_0.png"), str(save_folder / "images/Recons._0.png")


def test_wrapper(iter='ADMM', params=None):

    gt_loc, reconst_location = test_pnp(
        parameters_dict=params, iter=iter, BASE_DIR=Path(iter))
    return PSNR(gt_loc, reconst_location), iter, gt_loc, reconst_location


def tot_test(methods=['ADMM']):
    struct = {
        'method': [],
        'PSNR': [],
        'iter': [],
        'gt_loc': [],
        'reconstruct_loc': []
    }
    for method in methods:
        if method == 'DRS' or method == 'PGD':
            lamb, sigma_denoiser, stepsize, max_iter, alpha = get_ProxPnP_params(
                algo_name=method, noise_level_img=0.03)
            params = {
                "lambda": lamb,
                "g_param": sigma_denoiser,
                "stepsize": stepsize,
                "max_iter": max_iter,
                "alpha": alpha
            }

        elif method == 'DIPR':
            lamb, sigma_denoiser, stepsize, max_iter = get_DPIR_params(
                noise_level_img=0.03)
            params = {
                "lambda": lamb,
                "g_param": sigma_denoiser,
                "stepsize": stepsize,
                "max_iter": max_iter,
            }

        else:
            lamb, sigma_denoiser, stepsize, max_iter = get_GSPnP_params(
                problem="deblur", noise_level_img=0.03, k_index=0)
            params = {
                "lambda": lamb,
                "g_param": sigma_denoiser,
                "stepsize": stepsize,
                "max_iter": max_iter,
            }

        metric, iter, gt_loc, reconstruct_loc = test_wrapper(method, params)
        struct['method'].append(method)
        struct['PSNR'].append(metric)
        struct['iter'].append(iter)
        struct['gt_loc'].append(gt_loc)
        struct['reconstruct_loc'].append(reconstruct_loc)
    struct['matrix'] = difference_matrix(struct)
    return struct
# compare each mean_diff to eachoether and return a matrix


def difference_matrix(struct):
    diff_matrix = np.zeros((len(struct['method']), len(struct['method'])))
    for i in range(len(struct['method'])):
        for j in range(len(struct['method'])):
            diff_matrix[i][j] = difference_mean(
                struct['reconstruct_loc'][i], struct['reconstruct_loc'][j])
    return labels(diff_matrix, struct)


def labels(diff_matrix, struct):
    # make a table where struct are x and y labels
    df = pd.DataFrame(
        diff_matrix, columns=struct['method'], index=struct['method'])
    # df = df.iloc[0]
    return df


if __name__ == "__main__":
    methods = ['ADMM', 'PGD', 'DRS']
    struct = tot_test(methods)
    print(struct['matrix'])


# def find_variable_correlation(methods):
#     # try all \lambda values, attempt to make matrix == 0
