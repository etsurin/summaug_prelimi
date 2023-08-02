# SUMMAUG: An Effective Data Augmentation Method for Text Classification Task

This is the repository for summaug, the final course report of DL basics, 2023 summer. 

## Set up the environment

run the following command to build up environment

```
pip install -r requirements.txt
```

## Generate augmented pseudo samples

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

The data is also available at [here](https://drive.google.com/file/d/1MSJob6uRjWEcNTs3WwHXSyNmomd2g9ps/view?usp=sharing), unzip them to the same path as the code files. 

## Train a classification model

To train with original data, run
```
python roberta_main.py
```

To train with AEDA augmented and SUMMAUG augmented data, run
```
python roberta_main.py --augfile aeda_aug
```
```
python roberta_main.py --augfile summ_aug
```
respectively

When training with part of the dataset, add 
```
--n_sample [The number of training samples]
```
 to the original command
### Tuning hyperparameters

You can change training hyperparameters by simply adding commands, refer to the main function of roberta_main.py for more information

## Check experiments results

run the following command

```
bash run_group.sh
```

You should get the same results as the report . 

