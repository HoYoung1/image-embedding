[![Build Status](https://travis-ci.org/elangovana/image-embedding.svg?branch=master)](https://travis-ci.org/elangovana/image-embedding)

# Image embedding
Feature extraction for images.. See [ImageEmbedding.ipynb](ImageEmbedding.ipynb) for more details


## Datasets
- [Caviar dataset](https://lorisbaz.github.io/caviar4reid.html)
- [Marker1501 dataset](https://github.com/Cysu/open-reid/tree/master/reid/datasets)

## Benchmarks
-  [A Systematic Evaluation and Benchmark for Person Re-Identification: Features, Metrics, and Datasets](https://arxiv.org/pdf/1605.09653.pdf)


## Prerequisites

1. Python 3.6
2. Install virtual env

## Set up

1. Install dependencies
    ```bash
     pip install -r src/requirements_dev.txt
    ```
    

## Custom training using resnet


1. Evalute caviar dataset
    ```bash
    export PYTHONPATH=src
    python src/experiment_train.py  --dataset Caviar --traindir tests/imagesCaviar --valdir tests/imagesCaviar --outdir /tmp --epochs 10 --batchsize 32
    ```

2. Evalute market 1501 dataset . 
    ```bash
    export PYTHONPATH=src
    python src/experiment_train.py  --dataset Market1501 --traindir tests/imagesCaviar --valdir tests/imagesCaviar --outdir /tmp --epochs 10 --batchsize 32
    ```


## Use  model to evaluate dataset

**Note** The images in the tests folder are sample only. Please download the full dataset from the respective public database

1. Evalute caviar dataset
    ```bash
    export PYTHONPATH=src
    python src/main_predict_evaluate.py --dataset Caviar --modelpath <model_path>  --rawimagesdir tests/imagesCaviar
    ```

2. Evalute market 1501 dataset . 
    ```bash
    export PYTHONPATH=src
    python src/main_predict_evaluate.py --dataset Market1501 --modelpath <model_path>  --rawimagesdir tests/imagesMarket1501
    ```


## Acknowledgements

- Gary B. Huang, Manu Ramesh, Tamara Berg, and Erik Learned-Miller.
Labeled Faces in the Wild: A Database for Studying Face Recognition in Unconstrained Environments.
University of Massachusetts, Amherst, Technical Report 07-49, October, 2007.

- Custom pictorial structures for re-identification
D. S. Cheng, M. Cristani, M. Stoppa, L. Bazzani, V. Murino
In British Machine Vision Conference (BMVC), 2011 
CAVIAR4REID dataset / video / bibtex

- Liang Zheng, Liyue Shen, Lu Tian, Shengjin Wang, Jingdong Wang, Qi Tian, "Scalable Person Re-identification: A Benchmark", IEEE International Conference on Computer Vision (ICCV), 2015.