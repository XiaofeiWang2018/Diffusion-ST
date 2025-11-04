# Cross-modal Diffusion Modelling for Super-resolved Spatial Transcriptomics
- This is the official repository of the paper "Cross-modal Diffusion Modelling for Super-resolved Spatial Transcriptomics" from **MICCAI 2024**

## 1. Environment
- Python >= 3.8
- Pytorch >= 2.0 is recommended
- opencv-python
- sklearn
- matplotlib


## 2. Train
Use the below command to train the model on Xenium [[Data Link]](https://huggingface.co/datasets/Zeiler123/C3-Diff/resolve/main/Xenium.zip).
```
    python ./super_res_train.py
```

## 3. Test
Use the below command to test the model on the database.
```
    python ./super_res_sample.py
```


## 5. Citation
If you find our work useful in your research or publication, please cite our work:
```
@inproceedings{wang2024cross,
  title={Cross-modal diffusion modelling for super-resolved spatial transcriptomics},
  author={Wang, Xiaofei and Huang, Xingxu and Price, Stephen and Li, Chao},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={98--108},
  year={2024},
  organization={Springer}
}
```

