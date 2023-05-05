import torch
import torch.nn as nn

class EncoderLSTM(nn.Module):
    
    # Encoder definition
    def __init__(self, vocabulary_size, embedding_size, num_layers = 1, bidirectional = False):
        
        super(EncoderLSTM, self).__init__()
        
        self.emb = nn.Embedding(vocabulary_size, embedding_size)
        self.lstm = nn.LSTM(input_size = embedding_size, hidden_size = embedding_size, 
                            num_layers = num_layers, batch_first = True, bidirectional = bidirectional)
        
    def forward(self, x):

        x = self.emb(x)
        output, hidden = self.lstm(x)
        return output, hidden
    
class DecoderLSTM(nn.Module):
    
    # Decoder definition (no attention)
    def __init__(self, vocabulary_size, embedding_size, num_layers = 1, bidirectional = False):

        super(DecoderLSTM, self).__init__()
        
        self.D = 2 if bidirectional else 1
        self.n_layers = num_layers

        
        self.emb = nn.Embedding(vocabulary_size, embedding_size)
        self.lstm = nn.LSTM(input_size = embedding_size, hidden_size = embedding_size, 
                            num_layers = num_layers, batch_first = True, bidirectional = bidirectional)
        self.relu = nn.ReLU(inplace=False)
        self.lin = nn.Linear(self.D * embedding_size, vocabulary_size)
        
    def forward(self, x, y, h):
        
        x = self.emb(x)
        x = self.relu(x)
        x, hidden = self.lstm(x, h) 
        output = self.lin(x) 
        
        return output, hidden
        
class AttentionDecoderLSTM(nn.Module):
    
    # Decoder definition (with attention)
    def __init__(self, vocabulary_size, embedding_size, seq_len, num_layers = 1, bidirectional = False, dropout_factor=0.1):

        super(AttentionDecoderLSTM, self).__init__()
        
        self.seq_len = seq_len
        self.emb_size = embedding_size
        self.D = 2 if bidirectional else 1
        self.n_layers = num_layers
        
        self.emb = nn.Embedding(vocabulary_size, embedding_size)
        self.dropout = nn.Dropout(dropout_factor)
        self.alignment = nn.Linear(embedding_size * (self.D * self.n_layers + 1), seq_len)
        self.softmax = nn.Softmax(dim=1)
        self.lstm = nn.LSTM(input_size = embedding_size * (self.D + 1), hidden_size = embedding_size, 
                            num_layers = num_layers, batch_first = True, bidirectional = bidirectional)
        self.relu = nn.ReLU(inplace=False)
        self.lin = nn.Linear(self.D * embedding_size, vocabulary_size)
        
    def forward(self, x, y, hid):

        h, c = hid
        # x: (batch, 1, emb), y: (batch, seq, D * emb), h: (D, batch, emb), c: (1, batch, emb)
        
        x = self.emb(x)
        x = self.dropout(x)
        # print(f'x: {x.size()}, y: {y.size()}, h: {h.size()}, c: {c.size()}')
        
        # alignment: (emb * 2) -> (seq)
        attention_weights = self.softmax(
            self.alignment(
                torch.cat((x, h.view(-1, 1, self.n_layers * self.D * self.emb_size)), dim=-1)
            )
        )
        # context_vector: lc(attention_weight, y) size: (1, batch, emb)
        context_vector = torch.bmm(attention_weights, y)
        context_vector = self.relu(context_vector)
        input_decoder = torch.cat((x, context_vector), dim=-1)
        output, hidden = self.lstm(input_decoder, (h, c)) # x: (emb_size)
        output = self.lin(output) 
        
        return output, hidden        
        
        