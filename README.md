# Extended navigation Model for Real Urban Navigation (RUN) with pre-trained word embeddings

## Abstract  
In the last few decades, there has been a boom in new research for natural language tasks. High investments led to huge collections of data and new neural network architectures started to tackle natural language processing   problems. The RUN dataset was collected to fill a gap in the field of natural language navigation (Paz-Argaman & Tsarfaty, 2019). It is based on OpenStreetMap, a free navigation tool which offers huge amounts of data linked to  raw language instructions. Due to its many details and noise, the RUN dataset is especially interesting for outdoor navigation agents. To tackle natural language navigation for the RUN dataset, Paz-Argaman & Tsarfaty (2019)  developed an encoder-decoder model, called CGAEW, with an attention mechanism, an entity abstraction module to handle out-of-vocabulary words, and a word-state-processor to keep track of the current position. In this thesis, I  customized their model and extended the embedding layer by adding pre-trained Word2Vec word embeddings which were trained on the Google News dataset. I compared the reported accuracy of CGAEW, a reproduced version with current Python packages and my customized model. I found that the pre-trained word embeddings did not improve the overall accuracy of the model significantly. Nonetheless, the customized model was faster in training and more stable with a smaller standard deviation.

### Dependencies

* [Pytorch](https://pytorch.org/) - Machine learning library for Python-related dependencies
* [Anaconda](https://www.anaconda.com/download/) - Anaconda includes all the other Python-related dependencies
* [ArgParse](https://docs.python.org/3/library/argparse.html) - Command line parsing in Python
* [Gensim](https://radimrehurek.com/gensim/) - Needed to load the Word2Vec word embeddings

### Installation
Below are installation instructions under Anaconda.
IMPORTANT: We use python 3.7.3

 - Setup a fresh Anaconda environment and install packages: 
 ```sh
# create and switch to new anaconda env
$ conda create -n RUN python=3.7.3
$ source activate RUN

# install required packages
$ pip install -r requirements.txt
```

### Instructions
 - Here are the instructions to use the code base:
 
##### Train Model:
 - To train the model with options, use the command line:
```sh
$ python train_model.py --options %(For the details of options)
$ python train_model.py [-h] [short_name_arg] %(For explanation on the commands)
```
 - An example of running:
 ```sh
$ python train_model.py -m1 map_1 -m2 map_2 -me  30 -do 0.9
```
The model will start training on map_1 and map_2 in 30 epochs and a dropout of 0.9. 
##### Test Model:
- When you trained on map_1 and map_2, testing would be on map_3. If you want to save the results, use "-sr". Also, you have to specify the path to the saved trained model with "-fp". The command could look like this:
```sh
$ python test_model.py -mt map_3 -sr -fp ./tracks/trackmodelname.pkl
```

### License
This software and data are released under the terms of the Apache License, Version 2.0.
