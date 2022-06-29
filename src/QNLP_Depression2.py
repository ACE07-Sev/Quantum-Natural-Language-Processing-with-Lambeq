from lambeq import BobcatParser, TketModel, remove_cups, IQPAnsatz, AtomicType, NumpyModel, spiders_reader
import lambeq.tokeniser.spacy_tokeniser
from pytket.extensions.qiskit import AerBackend
from discopy import Word
from discopy.rigid import Functor


# Function for replacing the unknown word(s) with <unk> token
def replace(box):
    if box.name not in known_words_list:
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


# initializing the parser
# parser = BobcatParser(model_name_or_path='C:/Users/elmm/Desktop/CQM/Model')
parser = spiders_reader

# initializing the web parser (uncomment only when using web parser and having a WSL interpreter)
# parser = WebParser()

# uncomment for when you want ot use user input
# input_sentence = input("Please input a sentence: ")

# initializing the Aer backend (uncomment only when using TketModel)
backend = AerBackend()
# backend configs
backend_config = {
    'backend': backend,
    'compilation': backend.default_compilation_pass(2),
    'shots': 8192
}

# Loading checkpoint model (220 epoch model)
# model = TketModel.from_checkpoint('C:/Users/elmm/Desktop/CQM/runs/Jun08_16-58-26_LT-ELMM-T-MY/model.lt', backend_config=backend_config)
model = NumpyModel.from_checkpoint('C:/Users/elmm/Desktop/CQM/runs/Jun19_00-59-09_LT-ELMM-T-MY/model.lt')

print(model.symbols)
# parsing the input (uncomment for using user input enabled feature)
# diagram = parser.sentence2diagram(input_sentence)

# training dataset
train_labels, train_data = read_data('bc_train_data.txt')
# validation dataset
dev_labels, dev_data = read_data('bc_dev_data.txt')
# test dataset
test_labels, test_data = read_data('bc_test_data.txt')

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
tokeniser = lambeq.SpacyTokeniser()

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

# parsing the dataset into sentence diagrams
train_diagrams = parser.sentences2diagrams(train_data)
dev_diagrams = parser.sentences2diagrams(dev_data)
test_diagrams = parser.sentences2diagrams(test_data)

# merging all diagrams into one
all_diagrams = train_diagrams + dev_diagrams + test_diagrams

# known symbols (330 words)
known_words = []
known_words_list = []
for i in range(len(all_diagrams)):
    known_words = [box.name for box in all_diagrams[i].boxes if isinstance(box, Word)]
    for j in range(len(known_words)):
        known_words_list.append(known_words[j])

print(known_words_list)
# new sample
new_diagram = parser.sentence2diagram('I am depressed .')

# initializing the replace functor
replace_functor = Functor(ob=lambda x: x, ar=replace)

# tokenized diagram
replaced_diag = replace_functor(new_diagram)
replaced_diag.draw()

# removing cups
diagram = remove_cups(replaced_diag)
diagram.draw()

# initializing the ansatz for generating the circuit (1 layer, 3 qubits) (output : 1 qubit)
ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 1, AtomicType.PREPOSITIONAL_PHRASE: 1}, n_layers=1, n_single_qubit_params=3)
# converting the diagram to a circuit
diagram = ansatz(diagram)
# Circuit show
diagram.draw()

# predicting on new input
prediction = model.forward([diagram])
print(prediction)
