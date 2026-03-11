
'''
Performing anomaly detection, generating segmentation masks and bounding boxes for detected anomalies.
Generate:
- Segmentation masks in:               ./output/masks/
- Images with anomaly frames in:       ./output/frames/
'''

import numpy as np
from PatchRes.PatchRes import PatchRes
import matplotlib.pyplot as plt
import torch
from .functions import random_select_images_in_one_folder
from PIL import Image
import os
from tqdm import tqdm
import matplotlib


# from SAM.ui_funcs import MainWindow

matplotlib.use('Agg')


device = "cpu"
if __name__ == "__main__":
    with torch.no_grad():
        # getting test data
        test_data, path_list = random_select_images_in_one_folder(
            data_folder='./data/images/', num=100, return_paths=True)
        
        stride = 5
        anomaly_threshold = 0.1
        hidden_size = 30
        window_size = 50

        print(
            f"parameters:\nwindow size: ({window_size},{window_size})\nreservoir size: {hidden_size}\nanomaly threshold: {anomaly_threshold}")

        # training by only normal data
        TR = PatchRes(hidden_size=hidden_size, stride=stride, window_size=[
            window_size, window_size], anomaly_threshold=anomaly_threshold)

        train_data = random_select_images_in_one_folder(
            "./data/normal/", num=20, rand_select=True)
        
        print("training...\n")
        features_list = []
        for i in tqdm(range(train_data.shape[0])):
            features_list.append(TR.fit(train_data[i].unsqueeze(0)))
        # TR.fit(train_data)
        torch.save(torch.cat(features_list), 'features.pth')

        print("testing...\n")
        for i in tqdm(range(test_data.shape[0])):
            # pixel seg mask
            img = test_data[i]
            # print("range: ", img.min(), img.max())
            ps_mask, _, _, _, _ = TR.predict(
                img.unsqueeze(0), mode="pixel_seg") # input must be [batch_size, H, W]
            ps_mask = ps_mask.squeeze(0)
            ps_mask_img = (ps_mask * 255).astype(np.uint8)

            ps_mask_img = Image.fromarray(ps_mask_img, 'L')

            # img with frames
            od_mask, _, _, _, _ = TR.predict(
                img.unsqueeze(0), mode="OD")

            od_mask = od_mask.squeeze(0)
            od_mask_img = (od_mask * 255).astype(np.uint8)

            od_mask_img = Image.fromarray(od_mask_img, 'L')
            img_name = os.path.basename(path_list[i])

            # save images
            plt.imshow(img, cmap="gray")
            plt.imshow(od_mask_img, cmap="Reds", alpha=0.3,
                    vmin=0, vmax=np.percentile(img, 70))
            plt.axis('off')
            plt.savefig(
                f"./output/frames/{img_name[:-4]}_frame_{hidden_size}_{stride}_({window_size},{window_size})_{anomaly_threshold}.png", bbox_inches='tight', pad_inches=-0.1)
            plt.imsave(
                f"./output/masks/{img_name[:-4]}_mask_{hidden_size}_{stride}_({window_size},{window_size}).png", ps_mask_img, cmap='plasma')
