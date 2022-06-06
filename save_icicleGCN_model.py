import os
import torch


# def save_model(args, model, optimizer, current_epoch):
#     out = os.path.join(args.model_path, "checkpoint_{}.tar".format(current_epoch))
#     state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
#     # state = {'net': , 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
#     torch.save(state, out)


def save_model(args, model, current_epoch):
    out = os.path.join(args.test_path,"checkpoint_{}.tar".format(current_epoch))
    state = model.state_dict()
    torch.save(state, out)
