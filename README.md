# Exploring-the-Representativity-of-Art-Paintings

Yingying Deng, Fan Tang, Weiming Dong, Chongyang Ma, Feiyue Huang,  Oliver Deussen, Changsheng Xu  <br>

## results presentation 
<p align="center">
<img src="https://github.com/diyiiyiii/Exploring-the-Representativity-of-Art-Paintings/blob/master/image/1.png" width="100%" height="100%">
</p>
(a)Paintings sorted according to representativity value, from largest to smallest; (b) Famous works from artist’s homepage. Paintings enclosed in a yellow box are famous works reserved for verifying the accuracy of our method; representativity of paintings against the blue background ranges from 0:6 to 1:0, representativity of paintings against the orange background ranges from 0:0 to 0:4. <br>

## Experiment
### Requirements
* python 3.6
* pytorch 1.3.0
 
### Style-Enhanced Art Paintings Representation
#### Training 

The training dataset is collected from [WIKIART](https://www.wikiart.org/)
You can download the pretrained [resnet and vgg](https://drive.google.com/file/d/15IIETn17Xgg9TpYacymWPv0NxqEHJMTt/view?usp=sharing)model.
```
python resnet_finetune_unify.py
```
#### Testing 
You can download our pretrained [artist classification](https://drive.google.com/file/d/1HHfg5a_4SHiH6FQAIvH9VXGwJld2HamS/view?usp=sharing) model to obtain image features directly.
```
python feature_generation.py
```
### Graph-Based Representativity Learning
If you just need calculate the representivity, you shoud download [KPL](https://drive.google.com/file/d/1BSW9W1Qb6MDMiFiHMXevqzBFTNd_bEGB/view?usp=sharing) files.
```
python anchor.py
```


### Reference
If you use our work in your research, please cite us using the following BibTeX entry ~ Thank you ^ . ^. Paper Link [pdf](https://ieeexplore.ieee.org/abstract/document/9167477)<br> 
```
@article{deng2020exploring,
  title={Exploring the Representativity of Art Paintings},
  author={Deng, Yingying and Tang, Fan and Dong, Weiming and Ma, Chongyang and Huang, Feiyue and Deussen, Oliver and Xu, Changsheng},
  journal={IEEE Transactions on Multimedia},
  year={2020},
  publisher={IEEE}
}
```
