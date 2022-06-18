import collections
import pickle
import warnings
import os
from random import shuffle

import spacy
from discopy.tensor import Tensor
from discopy import Word
from discopy.rigid import Functor

import matplotlib.pyplot as plt
import numpy as np
from numpy import random, unique
from lambeq import AtomicType, IQPAnsatz, remove_cups, NumpyModel, spiders_reader
from lambeq import BobcatParser
from lambeq import Dataset
from lambeq import QuantumTrainer, SPSAOptimizer
from lambeq import TketModel
from lambeq import SpacyTokeniser
from pytket.extensions.qiskit import AerBackend

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
spacy.load('en_core_web_sm')

BATCH_SIZE = 30
EPOCHS = 300
SEED = 0


# Function for replacing low occuring word(s) with <unk> token
def replace(box):
    if isinstance(box, Word) and dataset.count(box.name) < 1:
        return Word('unk', box.cod, box.dom)
    return box


# reading dataset from a .txt file
def read_data(filename):
    labels, sentences = [], []
    with open(filename) as f:
        for line in f:
            t = int(line[0])
            labels.append([t, 1 - t])
            sentences.append(line[1:].strip())
    return labels, sentences


# train/test split
# training dataset
train_labels, train_data = read_data('bc_train_data.txt')
# validation dataset
dev_labels, dev_data = read_data('bc_dev_data.txt')
# test dataset
test_labels, test_data = read_data('bc_test_data.txt')

labels = train_labels + dev_labels + test_labels
data = train_data + dev_data + test_data

pairs = list(zip(labels, data))
shuffle(pairs)
print(len(pairs))

N_EXAMPLES = len(pairs)

# Shuffling datasets
train_labels, train_data = zip(*pairs[:round(N_EXAMPLES * 0.4)])
dev_labels, dev_data = zip(*pairs[round(N_EXAMPLES * 0.4):round(N_EXAMPLES * 0.8)])
test_labels, test_data = zip(*pairs[round(N_EXAMPLES * 0.8):])

# training set words (with repetition)
train_data_string = ' '.join(train_data)
train_data_list = train_data_string.split(' ')
# validation set words (with repetition)
dev_data_string = ' '.join(dev_data)
dev_data_list = dev_data_string.split(' ')
# test set words (with repetition)
test_data_string = ' '.join(test_data)
test_data_list = test_data_string.split(' ')

# initializing spacy tokenizer
tokeniser = SpacyTokeniser()

print(train_data)

# tokenize for words with suffix
train_data = tokeniser.tokenise_sentences(train_data)
dev_data = tokeniser.tokenise_sentences(dev_data)
test_data = tokeniser.tokenise_sentences(test_data)

# merging the tokenized words back into a sentence
for i in range(len(train_data)):
    train_data[i] = ' '.join(train_data[i])

for i in range(len(dev_data)):
    dev_data[i] = ' '.join(dev_data[i])

for i in range(len(test_data)):
    test_data[i] = ' '.join(test_data[i])

print(train_data)

# dataset words (with repetition)
dataset = train_data_list + dev_data_list + test_data_list
# list of all unique words in the dataset
unique_words = unique(dataset)
# frequency for each unique word
counter = collections.Counter(dataset)
print(counter)

# initializing the replace functor
replace_functor = Functor(ob=lambda x: x, ar=replace)

# initializing the parser
# parser = BobcatParser(model_name_or_path='C:/Users/elmm/Desktop/CQM/Model')
parser = spiders_reader

# Bobcat Parsed Diagrams
# parsing the dataset into sentence diagrams (requires Bobcat model to run locally)
raw_train_diagrams = parser.sentences2diagrams(train_data)
raw_dev_diagrams = parser.sentences2diagrams(dev_data)
raw_test_diagrams = parser.sentences2diagrams(test_data)

# Web Parsed Diagrams (alternative for Bobcat)(requires WSL to run locally)
wp_train_diagrams = open("C:/Users/elmm/Desktop/CQM/train.pkl", "rb")
wp_train_diagrams = pickle.load(wp_train_diagrams)

wp_test_diagrams = open("C:/Users/elmm/Desktop/CQM/test.pkl", "rb")
wp_test_diagrams = pickle.load(wp_test_diagrams)

wp_dev_diagrams = open("C:/Users/elmm/Desktop/CQM/dev.pkl", "rb")
wp_dev_diagrams = pickle.load(wp_dev_diagrams)

# Tokenizing low occuring words in each dataset
for i in range(len(raw_train_diagrams)):
    raw_train_diagrams[i] = replace_functor(raw_train_diagrams[i])

for i in range(len(raw_dev_diagrams)):
    raw_dev_diagrams[i] = replace_functor(raw_dev_diagrams[i])

for i in range(len(raw_test_diagrams)):
    raw_test_diagrams[i] = replace_functor(raw_test_diagrams[i])

# sample sentence diagram (entry 1)
raw_train_diagrams[0].draw()

# merging all diagrams into one for checking the new words
raw_all_diagrams = raw_train_diagrams + raw_dev_diagrams + raw_test_diagrams

# removing cups (after performing top-to-bottom scan of the word diagrams)
train_diagrams = [remove_cups(diagram) for diagram in raw_train_diagrams]
dev_diagrams = [remove_cups(diagram) for diagram in raw_dev_diagrams]
test_diagrams = [remove_cups(diagram) for diagram in raw_test_diagrams]

# sample sentence diagram (entry 1)
train_diagrams[0].draw()

# initializing the ansatz for generating the circuit (1 layer, 3 qubits) (output : 1 qubit)
ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1, AtomicType.PREPOSITIONAL_PHRASE: 1}, n_layers=1, n_single_qubit_params=3)

# train/test circuits
train_circuits = [ansatz(diagram) for diagram in train_diagrams]
dev_circuits = [ansatz(diagram) for diagram in dev_diagrams]
test_circuits = [ansatz(diagram) for diagram in test_diagrams]

# sample circuit diagram
train_circuits[0].draw(figsize=(9, 12))

# all circuits
all_circuits = train_circuits + dev_circuits + test_circuits

# initializing the Aer backend only when using TketModel
backend = AerBackend()
# backend configs
backend_config = {
    'backend': backend,
    'compilation': backend.default_compilation_pass(2),
    'shots': 8192
}

# initializing the Numpy Quantum Model model (TKET simulates a NISQ system and can be time consuming, hence we will use Numpy)
# model = TketModel.from_diagrams(all_circuits, backend_config=backend_config)
model = NumpyModel.from_diagrams(all_circuits)

# defining loss and accuracy (binary cross entropy)
loss = lambda y_hat, y: -np.sum(y * np.log(y_hat)) / len(y)
acc = lambda y_hat, y: np.sum(np.round(y_hat) == y) / len(y) / 2

# initializing the trainer for the model
trainer = QuantumTrainer(
    model,
    loss_function=loss,
    epochs=EPOCHS,
    optimizer=SPSAOptimizer,    # Simultaneous Perturbation Stochastic Approximation optimizer
    optim_hyperparams={'a': 0.2, 'c': 0.06, 'A': 0.01 * EPOCHS},
    evaluate_functions={'acc': acc},
    evaluate_on_train=True,
    verbose='text',
    seed=0
)
# training dataset (Quantum batch optimization)
train_dataset = Dataset(train_circuits, train_labels, batch_size=BATCH_SIZE)
# validation dataset
val_dataset = Dataset(dev_circuits, dev_labels, shuffle=False)
# fitting the model (training on train and validation dataset)
trainer.fit(train_dataset, val_dataset, logging_step=12)

# plotting results
fig, ((ax_tl, ax_tr), (ax_bl, ax_br)) = plt.subplots(2, 2, sharex=True, sharey='row', figsize=(10, 6))
ax_tl.set_title('Training set')
ax_tr.set_title('Development set')
ax_bl.set_xlabel('Iterations')
ax_br.set_xlabel('Iterations')
ax_bl.set_ylabel('Accuracy')
ax_tl.set_ylabel('Loss')

colours = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
ax_tl.plot(trainer.train_epoch_costs, color=next(colours))
ax_bl.plot(trainer.train_results['acc'], color=next(colours))
ax_tr.plot(trainer.val_costs, color=next(colours))
ax_br.plot(trainer.val_results['acc'], color=next(colours))
plt.show()

# Model accuracy on test dataset
test_acc = acc(model(test_circuits), test_labels)
print('Test accuracy:', test_acc)
