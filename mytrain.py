# Open and read contents of json file.
import json
from nltk_utils import bag_of_words, tokenize, stem
import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

#all_words =dictionary
#tag= Intents

###Reading the contents of the json input file
with open('intents.json', 'r') as f:
    intents = json.load(f)


###Data preprocessing
Intents = []
xy = []
dictionary=[]

# Json file loaded is a list therefore loop through it 
for i in intents['intents']:

#Extract intent and create the Dictionary after tokenizing and claening the input
    Intent= i['tag']
    #create list of Intents on which model is trained
    Intents.append(Intent)
    for j in i['patterns']:
        # remove punctuations
        RE_PUNCTUATION = re.compile("([!?.,;-])")
        Input_text = re.sub(RE_PUNCTUATION, " ", j)
        w = tokenize(Input_text)
        # add to our words list
        print(w)
        dictionary.extend(w)
        # add to xy pair
        xy.append((w, Intent))

# Dictionary should have only unique words therfore we sort out the repeating words
dictionary= sorted(set(dictionary))
Intents = sorted(set(Intents))
print('ssss')
print(dictionary)
print('ssss')
print(len(xy), "patterns")
print(len(Intents), "tags:", Intents)
print(len(dictionary), "unique stemmed words:", dictionary)
 

### Train dataset PREPARATION

# Input is the Sentences/questions
X_train = []

# Output/labels is the Intent of behind the question
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, dictionary)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = Intents.index(tag)
    y_train.append(label)


X_train = np.array(X_train)
y_train = np.array(y_train)



###Hyper-parameters 
num_epochs = 500
batch_size = 8
learning_rate = 0.008
input_size = len(X_train[0])
hidden_size = 8
output_size = len(Intents)
print(input_size, output_size)


###creating a pytorch dataset
class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # to get a text and label pair by one call
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

### To get reproducability I have used torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.manual_seed(110)
### To get reproducability remove dataloader#s randomness
g = torch.Generator()
g.manual_seed(110)

train_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=0,worker_init_fn=seed_worker,generator=g)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)  


# Loss and optimizer
Loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

###Training loop
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = Loss(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": dictionary,
"tags": Intents
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
