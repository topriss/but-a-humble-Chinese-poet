#  -*- coding:utf-8 -*-

import os
from collections import Counter

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class poem_dataset_class(Dataset):
  def __init__(self, fname='data/poems.datatxt', has_test=False, split_ratio=0.8):
    super(poem_dataset_class, self).__init__()
    len_limit = {'short': 10, 'long': 512}
    invalid_ch = [u'_', u'(', u'（', u'《', u'[', u'*', u'{']
    sep_ch = [u'，', u'。', u'？', u'！']
    
    poems = []
    assert os.path.exists(fname)
    for line in open(fname, 'r', encoding='utf-8').readlines():
      _, _, content = line.strip().split('::')
      content = content.replace(u' ', u'')
      if len(content) < len_limit['short'] or len(content) > len_limit['long']:
        continue
      if any([ch in content for ch in invalid_ch]):
        continue
      for sep in sep_ch:
        # content = content.replace(sep, u'|')
        content = content.replace(sep, u'')
      poems.extend([x + u' ' for x in content.split(u'|') if len(x) > 0])
    
    word_cnt = Counter()
    for poem in poems:
      word_cnt.update(poem)
    word_cnt[u' '] = -1
    self.words = [x[0] for x in sorted(word_cnt.items(), key=lambda x: -x[1])]
    self.num_words = len(self.words)
    self.word2scalar = dict(zip(self.words, range(self.num_words)))
    self.filler = self.word2scalar[u' ']
    
    self.poems_vec = [list([self.word2scalar[word] for word in poem]) for poem in poems]
    
    # if has_test:
    # 	assert 0.5 < split_ratio < 1
    # 	split_pos = int(len(poems_vec) * split_ratio)
    # 	self.train_vecs = poems_vec[:split_pos]
    # 	self.test_vecs = poems_vec[split_pos:]
    # else:
    # 	self.train_vecs = poems_vec
    # 	self.test_vecs = []
    assert not has_test, 'spilt data is not supported for now'
  
  def __getitem__(self, index):
    poem = self.poems_vec[index]
    x = torch.Tensor(poem[:-1]).type(torch.long)
    y = torch.Tensor(poem[1:]).type(torch.long)
    return x, y
  
  def __len__(self):
    return len(self.poems_vec)
  
  def collate_fn(self, batch):
    max_len = max([len(sp[0]) for sp in batch])
    rst_batch = torch.zeros(2, len(batch), max_len, dtype=torch.long)
    for i, sp in enumerate(batch):
      pad_len = max_len - len(sp[0])
      rst_sp = [F.pad(x, [0, pad_len], 'constant', self.filler) for x in sp]
      for j in [0, 1]:
        rst_batch[j, i] = rst_sp[j]
    
    return rst_batch


if __name__ == '__main__':
  dataset = poem_dataset_class()
  loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=dataset.collate_fn)
  for data_iter in loader:
    callme = 1
  callme = 2
