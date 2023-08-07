# README

The objective of this assignment is to reuse the CustomResNet model and build it using pytorch lightning. This model was used to train CIFAR10 dataset  

###Training logs and diagrams: 

No mentions of accuracy in the logs. Pasting graph pics below

INFO:pytorch_lightning.accelerators.cuda:LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
INFO:pytorch_lightning.callbacks.model_summary:
   | Name           | Type               | Params
-------------------------------------------------------
0  | criterion      | CrossEntropyLoss   | 0     
1  | prep_layer     | Sequential         | 1.9 K 
2  | convblock1     | Sequential         | 74.0 K
3  | res_block1     | Sequential         | 295 K 
4  | layer2         | Sequential         | 295 K 
5  | convblock2     | Sequential         | 1.2 M 
6  | res_block2     | Sequential         | 4.7 M 
7  | maxPool2       | MaxPool2d          | 0     
8  | output_linear  | Linear             | 5.1 K 
9  | accuracy       | MulticlassAccuracy | 0     
10 | train_accuracy | MulticlassAccuracy | 0     
-------------------------------------------------------
6.6 M     Trainable params
0         Non-trainable params
6.6 M     Total params
26.292    Total estimated model params size (MB)
Training: 0it [00:00, ?it/s]


![Training Loss](https://github.com/sagawritescode/ERAAssignment12/assets/45040561/90017128-9c83-47a4-9222-4bf340933002)

![Training Accuracy](https://github.com/sagawritescode/ERAAssignment12/assets/45040561/6ac37ca8-7b84-4f24-9665-ca13a1b88c3f)

Misclassified images: 
![Screenshot 2023-08-07 at 2 14 05 PM](https://github.com/sagawritescode/ERAAssignment12/assets/45040561/b9cffe58-0370-438c-a89d-83a3ea9e0007)

Spaces [link](https://huggingface.co/spaces/samundarcodes101/ResnetLightning2/blob/main/app.py)
