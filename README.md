# Polyvore Outfit Recognition

This projects implements Deep Learning models to recognize outfits from the Polyvore dataset.

## Dataset description
[Polyvore.com](https://www.ssense.com/) is a popular fashion website, where user can create and upload outfit data. 
Polyvore dataset is formed by taking images from this. 
This dataset contains 21,889 outfits from polyvore.com, in which 17,316 are for training, 1,497 for validation and 3,076 for testing.

## Models
The models include:
1. Finetuning a model to update the last layer weights. [Link](/finetuned_Model.py)
2. Building a model from scratch. [Link](/model_fromScratch.py)

## Results
The finetuned model gave an accuracy of 61%. The accuracy was increased to 69% in the model built from scratch.
![Finetuned Model Accuracy](/learning_accuracy_finetuned.png)
![Model from Scratch Accuracy](/learning_accuracy_fromScratch.png)
