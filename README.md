[![Build Status](https://travis-ci.org/elangovana/image-embedding.svg?branch=master)](https://travis-ci.org/elangovana/image-embedding)

# Image embedding
Feature extraction for images.. See [ImageEmbedding.ipynb](ImageEmbedding.ipynb) for more details


## Datasets
- [Caviar dataset](https://lorisbaz.github.io/caviar4reid.html)
- [Marker1501 dataset](https://github.com/Cysu/open-reid/tree/master/reid/datasets)

## Benchmarks
-  [A Systematic Evaluation and Benchmark for Person Re-Identification: Features, Metrics, and Datasets](https://arxiv.org/pdf/1605.09653.pdf)

## Implementation details

We try to implement the paper *Almazan, Jon, et al. "Re-id done right: towards good practices for person re-identification." arXiv preprint arXiv:1801.05339 (2018).*
 
## Prerequisites

1. Python 3.7.4
2. Install virtual env

## Set up

1. Install dependencies
    ```bash
     pip install -r src/requirements_prod.txt
    ```
    

## Custom training using resnet

1. Evalute caviar dataset **TODO: Caviar dataset factory doesnt work, fix..** 
    ```bash
    export PYTHONPATH=src
    python src/experiment_train.py  --dataset CaviarFactory --traindir tests/imagesCaviar --valdir tests/imagesCaviar --outdir /tmp --epochs 10 --batchsize 32
    ```

2. Evalute market 1501 dataset . 
    ```bash
    export PYTHONPATH=src
    python src/experiment_train.py  --dataset Market1501TripletFactory --traindir tests/imagesMarket1501 --valdir tests/imagesMarket1501 --outdir /tmp --epochs 10 --batchsize 32  --learning_rate .0001 --tripletloss_margin 1000
    ```


## Evaluate a custom trained model

**Note** The images in the tests folder are sample only. Please download the full dataset from the respective public database


1. Evalute caviar dataset. **TODO: Caviar dataset factory doesnt work, fix..** 

    ```bash
    export PYTHONPATH=src
    python src/main_predict_evaluate.py --dataset CaviarFactory --modelpath <model_path>  --queryimagesdir tests/imagesCaviar
    ```

2. Evalute market 1501 dataset . 
    ```bash
    export PYTHONPATH=src
    python src/main_predict_evaluate.py --dataset Market1501Factory --modelpath <model_path>  --queryimagesdir tests/imagesMarket1501
    ```

## Evaluate a pretrained model 

1. Save a pretrained model dict into a file using PretrainedModelLoader.py and then run main_predict_evaluate.py 

    ```bash
    export model_path=/tmp/modeldir/
    mkdir -p /tmp/modeldir/
   
    #save pretrained model  
    export PYTHONPATH=src
    python src/PretrainedModelLoader.py --modelname resnet18 --modelpath ${model_path}  
 
    # evaluate pretrained model 
    python src/main_predict_evaluate.py --dataset Market1501Factory --modelpath ${model_path}  --queryimagesdir tests/imagesMarket1501
         
    
    ```

## Acknowledgements

- Almazan, Jon, et al. "Re-id done right: towards good practices for person re-identification." arXiv preprint arXiv:1801.05339 (2018).

- Gary B. Huang, Manu Ramesh, Tamara Berg, and Erik Learned-Miller.
Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments.
University of Massachusetts, Amherst, Technical Report 07-49, October, 2007.

- Custom pictorial structures for re-identification
D. S. Cheng, M. Cristani, M. Stoppa, L. Bazzani, V. Murino
In British Machine Vision Conference (BMVC), 2011 
CAVIAR4REID dataset / video / bibtex

- Liang Zheng, Liyue Shen, Lu Tian, Shengjin Wang, Jingdong Wang, Qi Tian, "Scalable Person Re-identification: A Benchmark", IEEE International Conference on Computer Vision (ICCV), 2015.

- Triplet Loss and Online Triplet Mining in TensorFlow - https://omoindrot.github.io/triplet-loss
