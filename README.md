## Geography-Aware Self-Supervised Learning (ICCV 2021)
[**Project**](https://geography-aware-ssl.github.io/) | [**Paper**](https://arxiv.org/abs/2011.09980) | [**Poster**](https://geography-aware-ssl.github.io/static/images/ICCV%202021%20Poster.png)



[Kumar Ayush](https://kayush95.github.io)<sup>\*</sup>, [Burak Uzkent](https://uzkent.github.io/)<sup>\*</sup>, [Chenlin Meng](https://cs.stanford.edu/~chenlin/)<sup>\*</sup>, [Kumar Tanmay](), [Marshall Burke](https://web.stanford.edu/~mburke/), [David Lobell](https://earth.stanford.edu/people/david-lobell), [Stefano Ermon](https://cs.stanford.edu/~ermon/).
<br> Stanford University
<br>In [ICCV](https://arxiv.org/abs/2011.09980), 2021.

<p align="center">
  <img src="https://raw.githubusercontent.com/sustainlab-group/geography-aware-ssl/main/.github/images/ap2.png" width="500">
</p>


This is a PyTorch implementation of [Geography-Aware Self-Supervised Learning](https://arxiv.org/abs/2011.09980). We use the the official implementation of <a href="https://github.com/facebookresearch/moco">MoCo-v2</a> for developing our methods.

### Data Format

Your data directory should be in the following format:

```
${Bird_root}
├── bird_1
│   ├── xxx.jpg  
│   ├── xxx.jpg
│   ├── ...
├── bird_2
│   ├── xxx.jpg  
│   ├── xxx.jpg
│   ├── ...
├── ...



${Satellite_root}
├── xxx.jpg  
├── xxx.jpg  
├── ...

```




### Self-Supervised Training

Similar to official implementation of MoCo-v2, this implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported.

To do self-supervised pre-training of a ResNet-50 model on fmow using our MoCo-v2+Geo+TP model in an 4-gpu machine, run:
```
python moco_fmow/main_pretrain.py \ 
    -a resnet50 \
    --lr 0.03 \
    --dist-url 'tcp://localhost:14653' --multiprocessing-distributed --moco-t 0.02 --world-size 1 --rank 0 --mlp -j 4 \
    --loss cpc --epochs 200 --batch-size 256 --moco-dim 128 --aug-plus --cos \
    --save-dir ${PT_DIR} \
    --bird_path", ${Path to the bird image folder} \ 
    --satellite_path, ${Path to the satellite image folder} \ 
    --bird_satellite_pair_csv, ${Path to the bird satellite pair csv} \

```


### Linear Classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 4-gpu machine, run:
```
python moco_fmow/main_lincls.py \
    -a resnet50 \
    --lr 1 \
    --dist-url 'tcp://localhost:14653' --multiprocessing-distributed --world-size 1 --rank 0 -j 4 \
    --pretrained=${PT_DIR} \
    --save-dir ${PTDIR}/lincls \
    --data ${Path to the data folder} --batch-size 256
```
### Models

Our pre-trained ResNet-50 models can be downloaded as following:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">epochs</th>
<th valign="bottom">model</th>
<!-- TABLE BODY -->
<tr><td align="left">MoCo-v2</td>
<td align="center">200</td>
<td align="center"><a href="https://zenodo.org/record/7379715/files/moco.pth.tar?download=1">download</a></td>
</tr>
<tr><td align="left">MoCo-v2-Geo</td>
<td align="center">200</td>
<td align="center"><a href="https://zenodo.org/record/7379715/files/moco_geo.pth.tar?download=1">download</a></td>
</tr>
</tr>
<tr><td align="left">MoCo-v2-TP</td>
<td align="center">200</td>
<td align="center"><a href="https://zenodo.org/record/7379715/files/moco_tp.pth.tar?download=1">download</a></td>
</tr>
<tr><td align="left">MoCo-v2+Geo+TP</td>
<td align="center">200</td>
<td align="center"><a href="https://zenodo.org/record/7379715/files/moco_geo%2Btp.pth.tar?download=1">download</a></td>
</tr>
</tbody></table>


### Transfer Learning Experiments
We use Retina-Net implementation from this <a href="https://github.com/yhenon/pytorch-retinanet">repository</a> for object detection experiments on xView. We use PSANet implementation from this <a href="https://github.com/hszhao/semseg">repository</a> for semantic segmentation experiments on SpaceNet.


### Citing
If you find our work useful, please consider citing:
```
@article{ayush2021geography,
      title={Geography-Aware Self-Supervised Learning},
      author={Ayush, Kumar and Uzkent, Burak and Meng, Chenlin and Tanmay, Kumar and Burke, Marshall and Lobell, David and Ermon, Stefano},
      journal={ICCV},
      year={2021}
    }
```


