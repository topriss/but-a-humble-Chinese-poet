#  -*- coding:utf-8 -*-
import os
import argparse

import tqdm
import numpy as np
import torch
import torch.nn as nn

from dataset import poem_dataset_class
from model import LSTM_with_embedding


def prob_sample(weight, top_n=4):
  idx = np.argsort(weight)[:-1]
  t = np.cumsum(weight[idx[:top_n]])
  s = np.sum(weight[idx[:top_n]])
  pos = np.searchsorted(t, s * np.random.rand(1))[0]
  
  return idx[pos]


def infer(model, dataset, start=u'月'):
  h = torch.zeros(model.num_layer, 1, model.hidden_size)
  c = torch.zeros(model.num_layer, 1, model.hidden_size)
  x = nn.functional.embedding(
    torch.full(size=(1, 1), fill_value=dataset.word2scalar[start], dtype=torch.long),
    model.emb_weight
  )
  if args.cuda:
    x, h, c = x.cuda(), h.cuda(), c.cuda()
  
  max_len = 33
  poem = [start]
  while True:
    x, (h, c) = model.lstm(x, (h, c))
    x_prob = np.squeeze(model.softmax(x).data.cpu().numpy())
    word = dataset.words[prob_sample(x_prob)]
    if word in [u' ', u']'] or len(poem) >= max_len:
      break
    poem.append(word)
    # print(poem)
  
  return poem


def main():
  dataset = poem_dataset_class()
  loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True, collate_fn=dataset.collate_fn)
  model = LSTM_with_embedding(num_word=dataset.num_words)
  if args.cuda:
    model = model.cuda()
  opt = torch.optim.Adam(list(model.parameters()))
  
  for ep in range(args.epoch):
    iter_data = tqdm.tqdm(enumerate(loader),
                          desc="epoch {}:\t".format(ep), total=len(loader), bar_format="{l_bar}{r_bar}")
    for i, data in iter_data:
      x, y = data
      if args.cuda:
        x, y = x.cuda(), y.cuda()
      y_pred = model(x)
      loss = torch.nn.functional.cross_entropy(y_pred.permute(0, 2, 1), y)
      
      opt.zero_grad()
      loss.backward()
      opt.step()
      
      if i % 100 == 0:
        state_dict = {
          "epoch":   ep,
          "iter":    i,
          "loss":    loss.item(),
          "example": infer(model, dataset)
        }
        iter_data.write(str(state_dict))
    
    for _ in range(4):
      print(infer(model, dataset))
  
  return


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--epoch", type=int, default=20, help="number of epochs")
  parser.add_argument("--batch", type=int, default=4, help="batch size")
  parser.add_argument("--cuda", action='store_true', help="whether to use cuda")
  args = parser.parse_args()
  args.cuda = args.cuda and torch.cuda.is_available()
  print(args)
  
  main()
