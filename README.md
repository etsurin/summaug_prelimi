# SUMMAUG: An Effective Data Augmentation Method for Text Classification Task

This is the repository for summaug, the final course report of DL basics, 2023 summer. 

## Set up the environment

run the following command to build up environment

```
pip install -r requirements.txt
```

## Generate augmented data

For [AEDA](https://aclanthology.org/2021.findings-emnlp.234/) method, run the following command 
```
python aeda.py
```
The implementation of AEDA is based on the source code at https://github.com/akkarimi/aeda_nlp

For SUMMAUG method, run the following command
```
python summ.py
```

After the two steps, you should have two txt files named summ_aug.txt and aeda.aug.txt in your path. 

## Training a classification model
