#  -*- coding:utf-8 -*-

import torch
import torch.nn as nn


class softmax_layer(nn.Module):
  def __init__(self, in_features, out_features):
    super(softmax_layer, self).__init__()
    self.linear = nn.Linear(in_features=in_features, out_features=out_features)
  
  def forward(self, x):
    x_out = self.linear(x)
    x_out = torch.nn.functional.softmax(x_out, dim=-1)
    return x_out


class LSTM_with_embedding(nn.Module):
  def __init__(self, num_word=None, embedding_size=128, hidden_size=128, num_layer=2):
    super(LSTM_with_embedding, self).__init__()
    assert num_word is not None
    self.num_word = num_word
    self.embedding_size = embedding_size
    self.hidden_size = hidden_size
    self.num_layer = num_layer
    
    self.emb_weight = nn.Parameter(torch.randn(num_word, embedding_size, dtype=torch.float32, requires_grad=True))
    self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layer, batch_first=True)
    self.softmax = softmax_layer(in_features=hidden_size, out_features=num_word)
  
  def forward(self, x):
    x_out = nn.functional.embedding(x, self.emb_weight)
    x_out, _ = self.lstm(x_out)
    x_out = self.softmax(x_out)
    
    return x_out


if __name__ == '__main__':
  model = LSTM_with_embedding(num_word=101, embedding_size=11, hidden_size=21, num_layer=3)
  param = list(model.parameters())
  callme = 1
