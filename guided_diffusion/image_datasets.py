import os

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch
from skimage.measure import shannon_entropy
from PIL import Image,ImageOps

def load_data(data_root,dataset_use,status,SR_times,gene_num
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param lr_data_dir:
    :param other_data_dir:
    :param hr_data_dir:
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if dataset_use=='Xenium':
        dataset= Xenium_dataset(data_root,SR_times,status,gene_num)
    elif dataset_use=='Visium':
        dataset= Visium_dataset(data_root,gene_num)
    elif dataset_use=='NBME':
        dataset= NBME_dataset(data_root,gene_num)

    return dataset

class Xenium_dataset(Dataset):
    def __init__(self, data_root,SR_times,status,gene_num):

        if status=='Train':
            sample_name=['01220101', '01220102', 'NC1', 'NC2', '0418']
            # sample_name = ['NC2']
        elif status=='Test':
            sample_name = ['01220201', '01220202']

        SR_ST_all=[]
        ### HR ST
        for sample_id in sample_name:
            sub_patches=os.listdir(data_root+'Xenium/HR_ST/extract/'+sample_id)
            for patch_id in sub_patches:
                if SR_times==10:
                    SR_ST=np.load(data_root+'Xenium/HR_ST/extract/'+sample_id+'/'+patch_id+'/HR_ST_256.npy')
                elif SR_times==5:
                    SR_ST = np.load(data_root + 'Xenium/HR_ST/extract/' + sample_id + '/' + patch_id + '/HR_ST_128.npy')
                SR_ST=np.transpose(SR_ST,axes=(2,0,1))
                SR_ST_all.append(SR_ST)
        SR_ST_all=np.array(SR_ST_all)
        Sum=np.sum(SR_ST_all,axis=(0,2,3))
        # gene_order=np.argsort(Sum)[::-1][0:gene_num]
        gene_order=np.load(data_root + 'gene_order.npy')[0:gene_num]
        self.SR_ST_all=SR_ST_all[:,gene_order,...].astype(np.float64) # (X,50,256,256)

        # Sum_gene = np.sum(SR_ST_all, axis=(0, 2, 3))
        # gene_coexpre=np.zeros(shape=(gene_num,gene_num))
        # for i in range(gene_num):
        #     for j in range(gene_num):
        #         gene_coexpre[i,j]=Sum_gene[i]/Sum_gene[j]
        # np.save(data_root + 'gene_coexpre.npy',gene_coexpre)

        ####### norm
        # Z=np.sum(self.SR_ST_all)
        # self.SR_ST_all = np.log((1 + self.SR_ST_all) / (Z))

        # for ii in range(self.SR_ST_all.shape[0]):
        #     Max=np.max(self.SR_ST_all[ii])
        #     Min=np.min(self.SR_ST_all[ii])
        #     self.SR_ST_all[ii]=(self.SR_ST_all[ii]-Min)/(Max-Min)

        for ii in range(self.SR_ST_all.shape[0]):
            for jj in range(self.SR_ST_all.shape[1]):
                if np.sum(self.SR_ST_all[ii, jj]) != 0:
                    Max=np.max(self.SR_ST_all[ii,jj])
                    Min=np.min(self.SR_ST_all[ii,jj])
                    self.SR_ST_all[ii,jj]=(self.SR_ST_all[ii,jj]-Min)/(Max-Min)

        ### spot ST
        spot_ST_all=[]
        for sample_id in sample_name:
            sub_patches = os.listdir(data_root + 'Xenium/spot_ST/extract/' + sample_id)
            for patch_id in sub_patches:
                spot_ST = np.load(data_root + 'Xenium/spot_ST/extract/' + sample_id + '/' + patch_id + '/spot_ST.npy')
                spot_ST = np.transpose(spot_ST, axes=(2, 0, 1))
                spot_ST_all.append(spot_ST)
        spot_ST_all = np.array(spot_ST_all)
        self.spot_ST_all = spot_ST_all[:, gene_order, ...].astype(np.float64)

        ####### norm
        # Z = np.sum(self.spot_ST_all)
        # self.spot_ST_all = np.log((1 + self.spot_ST_all) / (Z))

        # for ii in range(self.spot_ST_all.shape[0]):
        #     Max=np.max(self.spot_ST_all[ii])
        #     Min=np.min(self.spot_ST_all[ii])
        #     self.spot_ST_all[ii]=(self.spot_ST_all[ii]-Min)/(Max-Min)

        for ii in range(self.spot_ST_all.shape[0]):
            for jj in range(self.spot_ST_all.shape[1]):
                if np.sum(self.spot_ST_all[ii, jj]) != 0:
                    Max = np.max(self.spot_ST_all[ii,jj])
                    Min = np.min(self.spot_ST_all[ii,jj])
                    self.spot_ST_all[ii,jj] = (self.spot_ST_all[ii,jj] - Min) / (Max - Min)

        ### WSI 5120
        WSI_5120_all = []
        for sample_id in sample_name:
            sub_patches = os.listdir(data_root + 'Xenium/WSI/extract/' + sample_id)
            for patch_id in sub_patches:
                WSI_5120 = np.load(data_root + 'Xenium/WSI/extract/' + sample_id + '/' + patch_id + '/5120_to256.npy')
                max=np.max(WSI_5120)
                WSI_5120 = np.transpose(WSI_5120, axes=(2, 0, 1))
                WSI_5120_all.append(WSI_5120)
        self.WSI_5120_all = np.array(WSI_5120_all)

        ### WSI 320
        self.num_320=[]
        WSI_320_all = []
        for sample_id in sample_name:
            sub_patches = os.listdir(data_root + 'Xenium/WSI/extract/' + sample_id)
            for patch_id in sub_patches:
                WSI_320 = np.load(data_root + 'Xenium/WSI/extract/' + sample_id + '/' + patch_id + '/320_to16.npy')

                # WSI_320_to16=[]
                # for i in range(WSI_320.shape[0]):
                #     im = Image.fromarray(WSI_320[i])
                #     im = im.resize((16, 16))
                #     im = np.asarray(im)
                #     WSI_320_to16.append(im)
                #
                # WSI_320_to16_NEW=WSI_320_to16
                # times = int(np.floor(256 / WSI_320.shape[0]))
                # remaining = 256 % WSI_320.shape[0]
                # if times > 1:
                #     for k in range(times - 1):
                #         WSI_320_to16_NEW = WSI_320_to16_NEW + WSI_320_to16
                # if not remaining == 0:
                #     WSI_320_to16_NEW = WSI_320_to16_NEW + WSI_320_to16[0:remaining]
                # WSI_320_to16_NEW = np.array(WSI_320_to16_NEW)
                # entropy=[]
                # for i in range(WSI_320_to16_NEW.shape[0]):
                #     entropy.append(np.round(shannon_entropy(WSI_320_to16_NEW[i])))
                # entropy=np.array(entropy)
                # entropy_order=np.argsort(entropy)
                # WSI_320_to16_entropy=WSI_320_to16_NEW[entropy_order]
                # np.save(data_root + 'NBME/WSI/extract/' + sample_id + '/' + patch_id + '/320_to16.npy',WSI_320_to16_entropy)
                # print(patch_id)

                WSI_320 = np.transpose(WSI_320, axes=(0, 3,1,2))
                WSI_320_all.append(WSI_320)
        self.WSI_320_all = np.array(WSI_320_all)
        max_320=np.max(WSI_320)
        a=1



    def __len__(self):
        return self.WSI_320_all.shape[0]

    def __getitem__(self, index):
        return self.SR_ST_all[index], self.spot_ST_all[index], self.WSI_5120_all[index], self.WSI_320_all[index]


class Visium_dataset(Dataset):
    def __init__(self, data_root,gene_num):

        sample_name = ['0701', '0106']

        gene_order = np.load(data_root + 'gene_order.npy')[0:gene_num]
        ### spot ST
        spot_ST_all = []
        for sample_id in sample_name:
            sub_patches = os.listdir(data_root + 'Visium/spot_ST/extract/' + sample_id)
            for patch_id in sub_patches:
                spot_ST = np.load(data_root + 'Visium/spot_ST/extract/' + sample_id + '/' + patch_id + '/spot_ST.npy')
                spot_ST = np.transpose(spot_ST, axes=(2, 0, 1))
                spot_ST_all.append(spot_ST)
        spot_ST_all = np.array(spot_ST_all)
        self.spot_ST_all = spot_ST_all[:, gene_order, ...].astype(np.float64)



        ####### norm
        # Z = np.sum(self.spot_ST_all)
        # self.spot_ST_all = np.log((1 + self.spot_ST_all) / (Z))

        # for ii in range(self.spot_ST_all.shape[0]):
        #     Max=np.max(self.spot_ST_all[ii])
        #     Min=np.min(self.spot_ST_all[ii])
        #     self.spot_ST_all[ii]=(self.spot_ST_all[ii]-Min)/(Max-Min)

        for ii in range(self.spot_ST_all.shape[0]):
            for jj in range(self.spot_ST_all.shape[1]):
                if np.sum(self.spot_ST_all[ii,jj])!=0:
                    Max = np.max(self.spot_ST_all[ii, jj])
                    Min = np.min(self.spot_ST_all[ii, jj])
                    self.spot_ST_all[ii, jj] = (self.spot_ST_all[ii, jj] - Min) / (Max - Min)



        ### WSI 5120
        WSI_5120_all = []
        for sample_id in sample_name:
            sub_patches = os.listdir(data_root + 'Visium/WSI/extract/' + sample_id)
            for patch_id in sub_patches:
                WSI_5120 = np.load(data_root + 'Visium/WSI/extract/' + sample_id + '/' + patch_id + '/5120_to256.npy')
                max = np.max(WSI_5120)
                WSI_5120 = np.transpose(WSI_5120, axes=(2, 0, 1))
                WSI_5120_all.append(WSI_5120)
        self.WSI_5120_all = np.array(WSI_5120_all)


        ### WSI 320
        self.num_320 = []
        WSI_320_all = []
        for sample_id in sample_name:
            sub_patches = os.listdir(data_root + 'Visium/WSI/extract/' + sample_id)
            for patch_id in sub_patches:
                WSI_320 = np.load(data_root + 'Visium/WSI/extract/' + sample_id + '/' + patch_id + '/320_to16.npy')
                WSI_320 = np.transpose(WSI_320, axes=(0, 3, 1, 2))
                WSI_320_all.append(WSI_320)
        self.WSI_320_all = np.array(WSI_320_all)
        max_320 = np.max(WSI_320)
        a = 1

        a=1


    def __len__(self):
        return self.WSI_320_all.shape[0]

    def __getitem__(self, index):
        return  self.spot_ST_all[index], self.WSI_5120_all[index], self.WSI_320_all[index]


class NBME_dataset(Dataset):
    def __init__(self, data_root,gene_num):
        sample_name = os.listdir(data_root + 'NBME/spot_ST/extract/')

        gene_order = np.load(data_root + 'gene_order.npy')[0:gene_num]
        ### spot ST
        spot_ST_all = []
        for sample_id in sample_name:
            sub_patches = os.listdir(data_root + 'NBME/spot_ST/extract/' + sample_id)
            for patch_id in sub_patches:
                spot_ST = np.load(data_root + 'NBME/spot_ST/extract/' + sample_id + '/' + patch_id + '/spot_ST.npy')
                spot_ST = np.transpose(spot_ST, axes=(2, 0, 1))
                spot_ST_all.append(spot_ST)
        spot_ST_all = np.array(spot_ST_all)
        self.spot_ST_all = spot_ST_all[:, gene_order, ...].astype(np.float64)



        ####### norm
        # Z = np.sum(self.spot_ST_all)
        # self.spot_ST_all = np.log((1 + self.spot_ST_all) / (Z))

        # for ii in range(self.spot_ST_all.shape[0]):
        #     Max=np.max(self.spot_ST_all[ii])
        #     Min=np.min(self.spot_ST_all[ii])
        #     self.spot_ST_all[ii]=(self.spot_ST_all[ii]-Min)/(Max-Min)

        for ii in range(self.spot_ST_all.shape[0]):
            for jj in range(self.spot_ST_all.shape[1]):
                if np.sum(self.spot_ST_all[ii, jj]) != 0:
                    Max = np.max(self.spot_ST_all[ii, jj])
                    Min = np.min(self.spot_ST_all[ii, jj])
                    self.spot_ST_all[ii, jj] = (self.spot_ST_all[ii, jj] - Min) / (Max - Min)

        ### WSI 5120
        WSI_5120_all = []
        for sample_id in sample_name:
            sub_patches = os.listdir(data_root + 'NBME/WSI/extract/' + sample_id)
            for patch_id in sub_patches:
                WSI_5120 = np.load(data_root + 'NBME/WSI/extract/' + sample_id + '/' + patch_id + '/5120_to256.npy')
                max = np.max(WSI_5120)
                WSI_5120 = np.transpose(WSI_5120, axes=(2, 0, 1))
                WSI_5120_all.append(WSI_5120)
        self.WSI_5120_all = np.array(WSI_5120_all)



        ### WSI 320
        self.num_320 = []
        WSI_320_all = []
        for sample_id in sample_name:
            sub_patches = os.listdir(data_root + 'NBME/WSI/extract/' + sample_id)
            for patch_id in sub_patches:
                WSI_320 = np.load(data_root + 'NBME/WSI/extract/' + sample_id + '/' + patch_id + '/320_to16.npy')
                WSI_320 = np.transpose(WSI_320, axes=(0, 3, 1, 2))

                WSI_320_all.append(WSI_320)
        self.WSI_320_all = np.array(WSI_320_all)
        max_320 = np.max(WSI_320)



        a = 1

    def __len__(self):
        return self.WSI_320_all.shape[0]

    def __getitem__(self, index):
        return self.spot_ST_all[index], self.WSI_5120_all[index], self.WSI_320_all[index]
