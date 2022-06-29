# QNLP

A Quantum Natural Language Processing model inspired by Lambek formulation and implemented in Python using Lambeq Package by Quantinuum. The model is designed for binary
classification for depression sentiment analysis, running on NumpyModel Ideal Quantum Simulator for speed and accuracy, and utilizing Spiders_Reader parsing for creating
the diagrams.

# Instructions
QNLP_Depression.py is the file for creating the model and training it, QNLP_Depression is for extracting predictions from a pretrained model. The bc_data.txt files are the datasets and the pkl files are WebParser diagrams. The pkls should be commented and/or ignored for when using Spiders_Reader.

model.lt is a pre-trained version of the model with %100 accuracy on the training set, %87 on the validation set and %76.9 on the test set.
