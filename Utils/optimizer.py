import torch


def get_Adam_optimizer(model, learning_rate=2e-5):
    optimizer = torch.optim.Adam([
        {'params': model.module.parameters(), 'lr': learning_rate},
    ], weight_decay=5e-4)
    return optimizer