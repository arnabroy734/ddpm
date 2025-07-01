#!/bin/bash
mkdir -p data
curl -L -o data/celeba-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/jessicali9530/celeba-dataset
cd data
unzip celeba-dataset.zip -d celeb-dataset
rm celeba-dataset.zip