import torch
import torch.nn as nn
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

criterion = nn.BCEWithLogitsLoss()
output = torch.tensor([-100.0, 0.5])
label = torch.tensor([1.0, 0.0])
criterion.pos_weight = torch.tensor([2.0])

loss = criterion(output, label)
true_loss = -0.5 * (2*math.log(sigmoid(-100.0)) + math.log(1 - sigmoid(0.5))) 

print(loss)
print(true_loss)