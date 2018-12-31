#  -*- coding:utf-8 -*-
import os
import sys
import argparse

import tqdm
import numpy as np
import torch
import torch.nn as nn

from dataset import poem_dataset_class
from model import LSTM_with_embedding


def prob_sample(weight, top_n=10):
  idx = np.argsort(-weight)[:-1]  # big first
  t = np.cumsum(weight[idx[:top_n]])
  s = np.sum(weight[idx[:top_n]])
  pos = np.searchsorted(t, s * np.random.rand(1))[0]
  
  return idx[pos], idx[:top_n], t / s


def infer_head(model, dataset, start=[u'风', u'花', u'雪', u'月']):
  sentence = [infer(model, dataset, start=s, num_sentence=1) for s in start]
  return u' '.join(sentence)


def infer(model, dataset, start=u'月', num_sentence=4):
  h = torch.zeros(model.num_layer, 1, model.hidden_size)
  c = torch.zeros(model.num_layer, 1, model.hidden_size)
  x = nn.functional.embedding(
    torch.full(size=(1, 1), fill_value=dataset.word2scalar[start], dtype=torch.long),
    model.emb_weight.cpu()
  )
  if args.cuda:
    x, h, c = x.cuda(), h.cuda(), c.cuda()
  
  max_len = 5 * num_sentence
  poem = start
  while True:
    x, (h, c) = model.lstm(x, (h, c))
    x_prob = np.squeeze(model.softmax(x).data.cpu().numpy())
    i_select, i_all, cum_prob = prob_sample(x_prob, top_n=np.random.randint(50, 100))
    # word = dataset.words[i_select]
    while True:
      i_rand = i_all[np.random.randint(0, len(i_all))]
      if i_rand < 3000:  # filter out uncommon word
        break
    word = dataset.words[i_rand]
    # print(u"\'{}\' from {}".format(word, [dataset.words[i] + ':' + str(cp) for i, cp in zip(i_all, cum_prob)]))
    if word in [u' ', u']'] or len(poem) >= max_len:
      break
    poem += word
  
  sys.stdout.flush()
  
  # insert ' ' for readability
  pos = 5
  while pos < len(poem):
    poem = poem[:pos] + u' ' + poem[pos:]
    pos += 6
  return poem


def main():
  # data
  dataset = poem_dataset_class(load_reduced=args.just_infer)
  if not args.just_infer:
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=6,
                                         collate_fn=dataset.collate_fn)
  # model
  model = LSTM_with_embedding(num_word=dataset.num_words)
  if os.path.exists(args.load):
    print("loading checkpoint \'{}\'".format(args.load))
    model.parameterize()
    model.load_state_dict(torch.load(args.load, map_location=torch.device('cpu')))
  if os.path.exists(args.pre_emb):
    print("loading embedding \'{}\'".format(args.pre_emb))
    with open(args.pre_emb, 'r', encoding='utf-8') as femb:
      emb_dim = int(femb.readline().strip().split()[1])
      assert emb_dim == model.embedding_size
      for line in femb.readlines():
        emb = line.strip().split()
        word = emb.pop(0)
        if word in dataset.word2scalar:
          model.emb_weight[dataset.word2scalar[word]] = torch.tensor(list(map(float, emb)), dtype=torch.float)
    model.parameterize()
  if args.cuda:
    model = model.cuda()
  
  # just infer
  if args.just_infer:
    while True:
      print(infer(model, dataset))
      print(infer_head(model, dataset))
      print()
      callme = 1
    return
    
    # schedule
  opt = torch.optim.Adam(list(model.parameters()), lr=args.lr)
  
  # main loop
  for ep in range(args.epoch):
    iter_data = tqdm.tqdm(enumerate(loader),
                          desc="epoch {}:\t\t".format(ep), total=len(loader), bar_format="{l_bar}{r_bar}")
    ep_loss = []
    best_loss = 123456789.0
    for i, data in iter_data:
      x, y = data
      if args.cuda:
        x, y = x.cuda(), y.cuda()
      y_pred = model(x)
      loss = torch.nn.functional.cross_entropy(y_pred.permute(0, 2, 1), y)
      
      opt.zero_grad()
      loss.backward()
      opt.step()
      
      if i % 1000 == 0:
        state_dict = {"epoch":   ep, "iter": i, "loss": loss.item(),
                      "example": infer(model, dataset)}
        iter_data.write(str(state_dict))
      ep_loss.append(loss.item())
    
    ep_loss = np.mean(ep_loss)
    if best_loss > ep_loss:
      best_loss = ep_loss
      if ep > 3:  # just to protect old checkpoint
        torch.save(model.state_dict(), './checkpoint/best.ckp')
    
    torch.save(model.state_dict(), './checkpoint/routine.ckp')
    for _ in range(4):
      print(infer(model, dataset))
  
  return


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--epoch", type=int, default=10000, help="number of epochs")
  parser.add_argument("--batch", type=int, default=256, help="batch size")
  parser.add_argument("--cuda", action='store_true', help="whether to use cuda")
  parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
  parser.add_argument("--load", type=str, default='./checkpoint/first_success/best.ckp', help="check point to load")
  parser.add_argument('--pre_emb', type=str, default='./data/wordembedding.datatxt',
                      help='pre-trained embedding to load')
  parser.add_argument('--just_infer', action='store_true', help='whether to just infer')
  
  args = parser.parse_args()
  
  # --- WARNING --- just for debug
  args.pre_emb = 'does not exist'
  args.just_infer = True
  
  args.cuda = args.cuda and torch.cuda.is_available() and (not args.just_infer)
  if args.just_infer:
    print('--- MODE ---: just infer')
  
  print(args)
  
  main()
