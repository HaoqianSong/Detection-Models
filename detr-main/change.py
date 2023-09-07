import torch
# 修改路径 预训练模型
pretrained_weights=torch.load('detr-r50.pth')
# 修改自己的类别
num_classes=1
pretrained_weights["model"]["class_embed.weight"].resize_(num_classes+1,256)
pretrained_weights["model"]["class_embed.bias"].resize_(num_classes+1)
torch.save(pretrained_weights,"detr_r50_%d.pth"%num_classes)
