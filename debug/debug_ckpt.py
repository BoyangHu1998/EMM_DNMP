import torch


path = 'training_outputs/chpt/render-dtu-0.001-0.005-test-point/ckpt_1000.pth'


ckpts = torch.load(path)['state_dict']

for key in ckpts.keys():
    print(key, ': ' ,ckpts[key].shape)
    print()