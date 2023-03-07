import torch.nn as nn

class EncoderLSTM(nn.Module):
    
    def __init__(self, vocabulary_size, embedding_size, num_layers = 1, bidirectional = False):
        
        super(EncoderLSTM, self).__init__()
        
        self.emb = nn.Embedding(vocabulary_size, embedding_size)
        self.lstm = nn.LSTM(input_size = embedding_size, hidden_size = embedding_size, 
                            num_layers = num_layers, batch_first = True, bidirectional = bidirectional)
        

        
    def forward(self, x):#, prev_hidden):
        
        x = self.emb(x)
        output, hidden = self.lstm(x)#, prev_hidden)
        
        return output, hidden
    
class DecoderLSTM(nn.Module):
    
    def __init__(self, vocabulary_size, embedding_size, num_layers = 1, bidirectional = False):

        super(DecoderLSTM, self).__init__()
        
        self.emb = nn.Embedding(vocabulary_size, embedding_size)
        self.lstm = nn.LSTM(input_size = embedding_size, hidden_size = embedding_size, 
                            num_layers = num_layers, batch_first = True, bidirectional = bidirectional)
        self.relu = nn.ReLU(inplace=False)
        
    def forward(self, x, prev_hidden):
        
        x = self.relu(x)
        output, hidden = self.lstm(x, prev_hidden)
        
        return output, hidden
        
        
        
        
        
        
        
        