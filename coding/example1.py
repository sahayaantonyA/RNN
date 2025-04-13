# IMPORT NECESSARY LIBRARIES 
import torch
import torch.nn as nn
import torch.optim as optim
# import torchtext
# from torchtext.vocab import GloVe
from torch.utils.data import DataLoader, Dataset
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from tqdm import tqdm

'''
WHY WE IMPORT THOSE LIBRARIES?

torch → Core PyTorch library.

torch.nn → Contains neural network components (like RNN, Linear layers, etc.).

torch.optim → Optimization algorithms (SGD, Adam, etc.).

torchtext.vocab.GloVe → Pre-trained word embeddings to improve text understanding.

nltk.tokenize.word_tokenize → Converts sentences into word tokens.

numpy → General numerical operations.

tqdm → Creates a nice progress bar for loops.
'''

# TRAINING DATA
train_data = [
    ("I love this movie", 1),
    ("The film was great", 1),
    ("An amazing experience", 1),
    ("I hate this movie", 0),
    ("The film was terrible", 0),
    ("Worst experience ever", 0),
]


# TOKENIZATION & VOCABULARY
nltk.download('punkt_tab')

TEXT = [word_tokenize(sentence.lower()) for sentence,_ in train_data]
vocab = set(word for sentence in TEXT for word in sentence)
word2idx = {word : idx + 1 for idx,word in enumerate(vocab)}
word2idx['<PAD>'] = 0

'''
Tokenization converts sentences into words (e.g., "I love this" → ["i", "love", "this"]).

Lowercasing ensures case-insensitive comparison.

vocab stores all unique words.

word2idx maps each word to a unique index (needed for model input).

Padding (<PAD>) ensures that all sentences have the same length.
'''

# CONVERT SENTENCE TO NUMERICAL REPRESENTATION
def encode_sentence(sentence):
    return torch.tensor([word2idx[word] for word in word_tokenize(sentence.lower())],dtype= torch.long)

x_train = [encode_sentence(sentence) for sentence, _ in train_data]
y_train = torch.tensor([label for _, label in train_data],dtype= torch.long)

'''
We convert words into numbers so the model can process them.

encode_sentence() converts words into their corresponding indices.

X_train stores numerical representations of all sentences.

y_train stores labels (0 or 1).
'''

# CREATE A DATASET CLASS & DATALOADER
class SentimentDataset(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
def pad_collate(batch):
    x, y = zip(*batch)
    max_len = max(len(seq) for seq in x)
    x_padded = [torch.cat([seq, torch.zeros(max_len - len(seq), dtype=torch.long)]) for seq in x]
    return torch.stack(x_padded), torch.tensor(y)

dataset = SentimentDataset(x= x_train, y= y_train)
dataloader = DataLoader(dataset= dataset, batch_size= 2, shuffle= True, collate_fn= pad_collate)


# DEFINE SIMPLE RNN MODEL
class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SentimentRNN, self).__init__()

        self.embedding = nn.Embedding(num_embeddings= vocab_size, embedding_dim= embedding_dim)
        self.rnn = nn.RNN(input_size= embedding_dim, hidden_size= hidden_dim, batch_first= True)
        self.fc = nn.Linear(in_features= hidden_dim, out_features= output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        return self.fc(hidden[-1])
    
'''
Embedding Layer converts word indices to dense vectors.

RNN Layer processes sequential data.

Fully Connected (Linear) Layer converts the final hidden state into a prediction.
'''

# INITIALIZE MODEL, LOSS, OPTIMIZER
model = SentimentRNN(vocab_size= len(word2idx), embedding_dim= 50, hidden_dim= 128, output_dim= 2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params= model.parameters(), lr= 0.001)

def train_model(model, dataloader, criterion, optimizer, epochs= 100):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in tqdm(dataloader, desc= f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1} , Loss {total_loss:.4f}")

train_model(model= model, dataloader= dataloader, criterion= criterion, optimizer= optimizer)

'''
optimizer.zero_grad() clears previous gradients.

output = model(X_batch) runs the forward pass.

loss.backward() computes gradients.

optimizer.step() updates weights.
'''


# EVALUATE THE MODEL
def predict_sentiment(model: SentimentRNN, sentence):
    model.eval()
    with torch.no_grad():
        x = encode_sentence(sentence= sentence).unsqueeze(0)
        output = model(x)
        prediction = torch.argmax(output,dim=1).item()
        return "Positive" if prediction == 1 else "Negative"
    
print(predict_sentiment(model, "I love this movie"))
print(predict_sentiment(model, "Worst movie ever")) 