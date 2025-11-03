# Self-Supervised SAR Despeckling by Integrating Denoiser Prior with Attributed Scattering Center(SARDIDP_ASC)

This repository is the official PyTorch implementation of SARDIDP_ASC.

---

> Deep learning has become the mainstream approach for SAR image despeckling. However, the lack of speckle-free SAR images poses a significant challenge for supervised deep learning methods. To overcome this limitation, we propose a self-supervised SAR despeckling method that leverages a denoiser prior and attributed scattering centers to enhance the training process. Specifically, we use the output of an external denoiser as a pseudo-label for despeckling, while spatially correlated speckle noise in SAR images is decorrelated through random downsampling. The network is then updated by optimizing the similarity between its output and the pseudo-label. Additionally, an attributed scattering center map is introduced to help the network recognize strong scatterers and better preserve image details. Experiments on both synthetic and real SAR datasets demonstrate that our method outperforms existing despeckling approaches.

## Requirements
- pip install -r requirements.txt

## Training process:
The first step: Use the visible light dataset to train a denoiser with good performance. The training instructions are as follows
```bash
 python train_single.py
```
---

Then in the second step, use the pre-trained model from the first step to assist in the training of SAR images.
```bash
python SAR_train_with_wjdata12.py
```
if you want to test the performence of the model, use SAR_test.py please.
## Dataset Preparation :
- BSD300 datasets can be download [here](https://pan.baidu.com/s/1VzAKtM-6uERQSUDVGUIzXg?pwd=1111).
- SAR_data can be download [here](https://pan.baidu.com/s/1Xi1CSN75sCT66gop9dr4uw?pwd=1111).

