"""
Generate a large batch of samples from a super resolution model, given a batch
of samples from a regular model from image_sample.py.
"""

import matplotlib.pyplot as plt
import argparse
import yaml
import torch.distributed as dist
import statistics
from torch.utils.data import DataLoader
from guided_diffusion.image_datasets import load_data
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    sr_create_model_and_diffusion,
    get_psnr,
    get_ssim,
    add_dict_to_argparser,
)
import scipy
import torch.nn.functional as F
from compare import *
from skimage.metrics import structural_similarity
import numpy as np
def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    print(args)
    logger.configure()

    logger.log("loading data...")

    data_num = 84 if args.dataset_use =='Xenium' else (47 if args.dataset_use =='Visium' else 223)
    data = load_superres_data(args.batch_size,args.data_root, args.dataset_use, status='Test', SR_times=args.SR_times,
                                       gene_num=args.gene_num)



    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(args)
    
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    
    model.to(dist_util.dev())
    
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()


    logger.log("creating samples...")
    ############################
    rmse_all= np.zeros(shape=(data_num,args.gene_num))
    ssim_all = np.zeros(shape=(data_num,args.gene_num))
    cc_all = np.zeros(shape=(data_num,args.gene_num))


     
    for i in range(data_num // args.batch_size):
        if args.dataset_use=='Xenium':
            hr, model_kwargs = next(data)
            if args.SR_times==5:
                hr=F.interpolate(hr, size=(256, 256))
            hr = hr.permute(0, 2, 3, 1)
            hr = hr.contiguous()
            hr = hr.cpu().numpy()
        elif args.dataset_use=='Visium' or args.dataset_use=='NBME':
            model_kwargs = next(data)
            if args.dataset_use=='NBME':
                aaa=model_kwargs['low_res']
                model_kwargs['low_res']=F.interpolate(aaa, size=(26, 26))
            hr =model_kwargs['low_res']
            hr = hr.permute(0, 2, 3, 1)
            hr = hr.contiguous()
            hr = hr.cpu().numpy()

        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}

        # Choose the appropriate sample function based on the sampling method
        sample_fn = diffusion.ddim_sample_loop if args.sampling_method == 'ddim' else \
            (diffusion.dpm_solver_sample_loop if args.sampling_method == 'dpm++' else diffusion.p_sample_loop)

        sample = sample_fn(
            model,
            (args.batch_size, args.gene_num, model_kwargs['WSI_5120'].shape[2], model_kwargs['WSI_5120'].shape[3]),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )



        sample = sample.permute(0, 2, 3, 1)
        sample = sample.cpu().numpy() #[1,256,256,Gene]

        if args.dataset_use == 'Xenium':
            for gene in range(hr.shape[3]):
                GT = hr[0, ..., gene]
                pred = sample[0, ..., gene]
                cc_ours = scipy.stats.pearsonr(np.reshape(pred, (-1)), np.reshape(GT, (-1)))
                cc_all[i, gene] = abs(cc_ours.statistic)
                GT = (GT - np.min(GT)) / (np.max(GT) - np.min(GT))
                pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
                denom=GT.max() - GT.min()
                mse=np.mean((pred - GT) ** 2, dtype=np.float64)
                rmse = np.sqrt(mse) / denom
                rmse_all[i,gene]=rmse
                ssim = get_ssim(np.uint8(pred*225), np.uint8(GT*225))
                ssim_all[i,gene]=ssim

        elif args.dataset_use == 'Visium' or args.dataset_use == 'NBME':
            for gene in range(hr.shape[3]):
                GT = hr[0, ..., gene]
                pred = sample[0, ..., gene]

                GT = (GT - np.min(GT)) / (np.max(GT) - np.min(GT))
                pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
                pred[0, :] = np.min(pred)
                pred[:, 0] = np.min(pred)
                if np.mean(hr[0, ..., gene]) == 0:
                    continue
                maps = np.ones(shape=(26, 26))
                maps[GT == 0] = 0
                pred = pred * maps
                denom = GT.max() - GT.min()
                mse = np.mean((pred - GT) ** 2, dtype=np.float64)
                rmse = np.sqrt(mse) / denom
                rmse_all[i, gene] = rmse
                ssim = get_ssim(np.uint8(pred * 225), np.uint8(GT * 225))
                ssim_all[i, gene] = ssim
                cc_ours = scipy.stats.pearsonr(np.reshape(pred, (-1)), np.reshape(GT, (-1)))
                cc_all[i, gene] = abs(cc_ours.statistic)

                # print(gene)
                a = 1

        print('Ours: rmse_persample',np.mean(rmse_all[i,:]))
        print('Ours: ssim_persample', np.mean(ssim_all[i,:]))
        print('Ours: cc_persample', np.mean(cc_all[i,:]))
        print(i)


    dist.barrier()
    logger.log("sampling complete")

    print('Ours: rmse', np.mean(rmse_all[np.nonzero(rmse_all)]))
    print('Ours: ssim', np.mean(ssim_all[np.nonzero(ssim_all)]))
    print('Ours: cc', np.mean(cc_all[np.nonzero(cc_all)]))



def load_superres_data(batch_size,data_root,dataset_use,status,SR_times,gene_num):
    # Load the super-resolution dataset using the provided directories
    dataset = load_data(data_root=data_root,dataset_use=dataset_use,status=status,SR_times=SR_times,gene_num=gene_num)
    # Create a data loader to load the dataset in batches
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=False,
        pin_memory=True
    )
    if dataset_use=='Xenium':
        for SR_ST, spot_ST, WSI_5120, WSI_320 in loader:
            model_kwargs = {"low_res": spot_ST, "WSI_5120": WSI_5120, "WSI_320": WSI_320}
            yield SR_ST, model_kwargs
    elif dataset_use=='Visium' or dataset_use=='NBME':
        for spot_ST, WSI_5120, WSI_320 in loader:
            model_kwargs = {"low_res": spot_ST, "WSI_5120": WSI_5120, "WSI_320": WSI_320}
            yield model_kwargs



def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to YAML configuration file")
    args = parser.parse_args()

    # Load the configuration from the YAML file
    with open('config/config_test.yaml', "r") as file:
        config = yaml.safe_load(file)

    # Add the configuration values to the argument parser
    add_dict_to_argparser(parser, config)

    return parser


if __name__ == "__main__":
    main()
    # guided_DM()
