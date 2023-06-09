# A Flexible and Efficient Approach for Key Information Extraction

This repository contains the Jupyter Notebooks with code for comparing the BERT-base, RoBERTa-base, and LayoutLM v1 models with the XGBoost Classifier model on the SROIE dataset for the token classification task. It also has the preprocessed dataset with pairs of token and its label according to the classes defined in the SROIE dataset. The repository is related to the master thesis project:

_A Flexible and Efficient Approach for Key Information Extraction_

Author: Farkhad Kuanyshkereyev

Supervisor: Phuong T. Nguyen

Co-Supervisor: Henri Lagerroos

## Introduction

Current state-of-the-art approaches for the Key Information Extraction mostly include deep learning algorithms. They are known for high memory requirements and longer time for training the models and performing inference with them. Moreover, deep learning models usually have a pre-defined input features, which requires the user to perform proper feature extraction.

This project proposes an approach that has a comparable performance on the SROIE dataset with the state-of-the art models and provides faster training and testing time along with higher flexibility for feature engineering. SROIE is a Scanned Receipts OCR and Information Extraction dataset that has approximately 1000 of receipt images with bounding boxes for each word and 4 labels of extracted information - Address, Company Name, Date, and Total. This dataset is frequently used for benchmarking on the Key Information Extraction task.

The approach presented here is a flexible framework. Different sets of features could be provided to different pre-trained models for encoding them as embedding vectors. These embedding vectors could be fused together (through concatenation or any other operation) and then provided to the light-weight machine learning classification algorithm.

This project obtained currently the best combination of the framework elements that produces the highest entity-level F1 score on the SROIE testing dataset. The combination comprises visual, textual, and positional features, where visual and textual features are encoded using the ResNet-18 and MPNet pre-trained models, and the classification model with the hyperparameter tuning was the XGBoost classifier.

<p align="center">
<img src="https://github.com/Faranio/master-thesis-code/blob/master/images/ProposedFramework.png" width="550">
</p>

Different combinations of the framework elements are presented in the paper. Certain metrics such as the elapsed time for training and testing, memory size, entity-level precision, recall, and F1 score are recorded for each experiment. These results were compared with the baseline algorithms such as BERT-base, RoBERTa-base, and LayoutLM v1 in the paper.

The full SROIE dataset can be accessed through the following [link](https://www.kaggle.com/datasets/urbikn/sroie-datasetv2).

## Repository Structure

This repository is organized as follows:

* The [dataset](./dataset) directory contains preprocessed SROIE dataset that could be used for training the BERT-base, RoBERTa-base, and LayoutLM v1 algorithms:
  * [labels.txt](./dataset/labels.txt): Set of labels used in the token classification that are present in the SROIE dataset 
  * [train.txt](./dataset/train.txt): Pairs of token and a corresponding label for each of the images in the SROIE training set, separated by an empty line for each image
  * [test.txt](./dataset/test.txt): Pairs of token and a corresponding label for each of the images in the SROIE testing set, separated by an empty line for each image
  * [train_box.txt](./dataset/train_box.txt): Pairs of token and a corresponding upper-left and lower-right x and y coordinates of the token (x1, y1, x2, y2) for each of the images in the SROIE training set, separated by an empty line for each image
  * [test_box.txt](./dataset/test_box.txt): Pairs of token and a corresponding upper-left and lower-right x and y coordinates of the token (x1, y1, x2, y2) for each of the images in the SROIE testing set, separated by an empty line for each image
  * [train_image.txt](./dataset/train_image.txt): Rows of token, bounding box coordinates normalized to 1000x1000, width and height of the image, and the name of the image from the SROIE training set, separated by an empty line for each image
  * [test_image.txt](./dataset/test_image.txt): Rows of token, bounding box coordinates normalized to 1000x1000, width and height of the image, and the name of the image from the SROIE testing set, separated by an empty line for each image
* The [notebooks](./notebooks) directory contains the Jupyter Notebooks with the implementation of algorithms used for comparing their metrics on the SROIE dataset:
  * [BERT.ipynb](./notebooks/BERT.ipynb): contains code for training the BERT-base and the RoBERTa-base algorithms and measuring their performance on the testing set.
  * [LayoutLM.ipynb](./notebooks/LayoutLM.ipynb): contains code for preparing the data and training the LayoutLM v1 and measuring its performance on the testing set.
  * [dataset_preparation.ipynb](./notebooks/dataset_preparation.ipynb): contains code for preprocessing the dataset for the light-weight machine learning classification model.
  * [training.ipynb](./notebooks/training.ipynb): contains code for training and testing the light-weight machine learning classification model on the preprocessed dataset.
* The [requirements.txt](./requirements.txt) file contains all the packages and their versions necessary to run the Jupyter Notebooks.

## Troubleshooting

If you encounter any difficulties in working with the tool or the datasets, please do not hesitate to contact me at the following email: farkhad.kuanyshkereyev@gmail.com. I will try my best to answer you as soon as possible.