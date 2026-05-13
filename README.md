# DSS-USOD
Source code for our paper **[Learning Dynamic Structural Specialization for Underwater Salient Object Detection](https://arxiv.org/abs/2506.19472)**.

Created by **Lin Hong**, email: eelinhong@ust.hk

> We are open to collaboration and are looking for self-motivated students. If you're interested in this topic, please feel free to contact me via email.

---

## Overview
The [trained model](https://pan.baidu.com/s/1TwwaTcdmTiU2FHOC5xC3Vw) (Baidu Netdisk, fetch code: ie0k) or [Google Drive (DSS-USOD baseline)](https://drive.google.com/file/d/1fFKhuuR2MEEjBWRtjFdMZCoRxL) can be downloaded.

![DSS-USOD Framework](framework7.png)

### Requirements
- Python 3.8
- PyTorch 1.6.0
- TorchVision 0.7.0

---

## Benchmark
We retrained 40 SOTA methods in the fields of SOD and USOD. Here is the qualitative evaluation of the 40 SOTA methods and the proposed DSS-USOD baseline.

![Qualitative Evaluation](qualitative_compare2.png)

### Available Resources
1. **Predicted saliency maps of USOD10K**  
   Baidu Netdisk: [Link](https://pan.baidu.com/s/1EpnE07lgamyaUIUZWdccqA) | Fetch code: usod  
   Google Drive: [Link](https://drive.google.com/file/d/1D4wLLol843DEpolmO-cYpo2jaiBY7Ufn/view?usp=drive_link)

2. **Predicted saliency maps of USOD**  
   Baidu Netdisk: [Link](https://pan.baidu.com/s/1cnmMZ0JSshssm2jc9p2BdA) | Fetch code: usod  
   Google Drive: [Link](https://drive.google.com/file/d/1YoXKUKaauy2PkkISpK-QWJpetXIsTsrO/view?usp=drive_link)

3. **Evaluation results**  
   Baidu Netdisk: [Link](https://pan.baidu.com/s/1AL4WQeFh1KrD0jj9JW182g) | Fetch code: usod  
   Google Drive: [Link](https://drive.google.com/file/d/1jCuCvK-UJYq3g_TMQ7NTWqXfXbG21bk/view?usp=drive_link)

---

## Bibliography Entry
If you think our work is helpful, please cite:

```bibtex
@ARTICLE{10102831,
  author={Hong, Lin and Wang, Xin and Zhang, Gan and Zhao, Ming},
  journal={IEEE Transactions on Image Processing},
  title={USOD10K: A New Benchmark Dataset for Underwater Salient Object Detection},
  year={2025},
  volume={34},
  number={},
  pages={1602-1615},
  doi={10.1109/TIP.2023.3266163}
}
```

---

## Acknowledgement
We thank the developer of [MMdetection](https://github.com/open-mmlab/mmdetection), [WaterMask](https://github.com/LiamLian0727/WaterMask), and [USIS-SAM](https://github.com/LiamLian0727/USIS10K) for providing their open-source code, which greatly facilitated our benchmark evaluations.

---

## Note to Active Participants
We hope this work offers a new perspective on RGB-based USOD by demonstrating the importance of explicitly disentangling and dynamically coordinating complementary structural representations learned from diverse underwater images. Your contributions and feedback are welcome!
