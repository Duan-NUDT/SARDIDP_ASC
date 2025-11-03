# SARDIDP_ASC


## Requirements
- pip install -r requirements.txt

## Training process:
The first step: Use the visible light dataset to train a denoiser with good performance. The training instructions are as follows

- python train_single.py

Then in the second step, use the pre-trained model from the first step to assist in the training of SAR images.

- python SAR_train_with_wjdata12.py

if you want to test the performence of the model, use SAR_test.py please.
## Dataset Preparation :
BSD300 datasets can be download [here](https://pan.baidu.com/s/1VzAKtM-6uERQSUDVGUIzXg?pwd=1111)

