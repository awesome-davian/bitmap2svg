import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
from Attention import Attn
import torch.nn.functional as F

#Stacked Attention Networks for Image Question Answering
class SANDecoder(nn.Module):
    def __init__(self,  feature_size, hidden_size, vocab_size, num_layers):
        super(SANDecoder, self).__init__()
        #Define parameters
        self.feature_size = feature_size

        #Define layers
        self.embed = nn.Embedding(vocab_size, feature_size)
        self.init_layer = nn.Linear(feature_size, hidden_size)
        self.attn = Attn('general', feature_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.ctx2out = nn.Linear(feature_size, feature_size)
        self.h2out = nn.Linear(hidden_size, feature_size)
        self.out = nn.Linear(feature_size, vocab_size)
        self.out_cat = nn.Linear(feature_size*3, vocab_size)
        self.second_attn = nn.Linear(hidden_size, feature_size)
    
    def decode_lstm(self, input_word, context, hidden, lstm_out):

        hidden = hidden.squeeze(0)
        out = self.h2out(hidden)
        context = context.squeeze(1)
        out += self.ctx2out(context)
        out += input_word
        out = F.tanh(out)
        out = self.out(out)
       
        return out

    def init_lstm(self, features):

        sums = torch.sum(features, 2)
        out = torch.mul(sums, 1/features.size(2))
        out = out.squeeze(2).unsqueeze(0) # 1, batch, feature_size
        out = self.init_layer(out.squeeze(0)).unsqueeze(0)
        out = F.tanh(out)

        return out, out 


    def forward(self, features, captions, lengths):

        max_length = max(lengths)
        embed = self.embed(captions)
        h, c= self.init_lstm(features)
        arr = [] 

        for i in range(max_length):            
            if i == 0 :
                input_word = Variable(torch.zeros(embed.size(0), embed.size(2))).cuda(1)
            else: 
                input_word = embed[:,i-1]
            context = self.attn(h, features)           
            lstm_input = torch.cat((context, input_word.unsqueeze(1)),2)

            u_vec = self.second_attn(lstm_input.squeeze(1))
            second_context = self.attn(u_vec.unsqueeze(0),features)
            new_lstm_input = torch.cat((second_context, input_word.unsqueeze(1)),2)

            lstm_out, (h,c) = self.lstm(new_lstm_input, (h,c))
            out = self.decode_lstm(input_word,context, h, lstm_out).unsqueeze(1)
            arr += [out]

        return torch.cat(arr,1)

    def sample(self, features):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        h,c = self.init_lstm(features)

        for i in range(100):                                      # maximum sampling length
            if i == 0:
                x = Variable(torch.rand(1,1,self.feature_size)).cuda(1)
            else: 
                x = self.embed((predicted))

            context = self.attn(h, features)
            lstm_input = torch.cat((context, x) ,2)
            u_vec = self.second_attn(lstm_input.squeeze(1))
            second_context = self.attn(u_vec.unsqueeze(0),features)
            new_lstm_input = torch.cat((second_context, x),2)
            
            lstm_out, (h,c) = self.lstm(new_lstm_input, (h,c))          # (batch_size, 1, hidden_size), 
            out = self.decode_lstm(x, context, h, lstm_out)
            predicted = out.max(1)[1]
            sampled_ids.append(predicted)
        #sampled_ids = torch.cat(sampled_ids, 1)                  # (batch_size, 20)
        return sampled_ids
