import jittor as jt
import pygmtools as pygm
import os
import numpy as np
from model import GMNet
from dataset import GraphPair
from utils.parser import make_parser
pygm.BACKEND = 'jittor'

def train(model, train_loader, optimizer, epoch_idx, file):
    model.train()
    batch_size = train_loader.batch_size
    num_data = len(train_loader)

    train_loss = []
    for batch_idx, (img1, img2, kpts1, kpts2, A1, A2, ids, labels) in enumerate(train_loader):
        # print(img1.shape)
        # print(img2.shape)
        # print(kpts1.shape)
        # print(kpts2.shape)
        # print(A1.shape)
        # print(A2.shape)
        # print(labels.shape)
        outputs = model(img1, img2, kpts1, kpts2, A1, A2)
        loss = pygm.utils.permutation_loss(outputs, labels)
        optimizer.step(loss)
        train_loss.append(loss.item())
        print(loss.item())
        if batch_idx % 100 == 0:
            file.write(
                "Train epoch: {}  {:.2f}%\tLoss:{:.6f}\n".format(
                    epoch_idx, 100 * batch_idx * batch_size / num_data, loss.item()
                )
            )
    return np.mean(train_loss)


def main(args):
    batch_size, learning_rate, weight_decay, num_epoch, debug = (args.bs, args.lr, args.wd, args.ne, args.debug)
    if debug:
        num_epoch = 1

    train_loader = GraphPair(sets="train", batch_size=batch_size)
    model = GMNet()
    optim = jt.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    folder_name = f"bs_{batch_size}_lr_{learning_rate}_wd_{weight_decay}_ne_{num_epoch}"
    if not os.path.exists(f"./checkpoint/{folder_name}"):
        os.mkdir(f"./checkpoint/{folder_name}")
    file_name = f"./checkpoint/{folder_name}/log.txt"
    file = open(file_name, "w")
    file.write(f"{folder_name}\n")
    train_losses, test_losses = [], []
    for epoch_idx in range(1, num_epoch + 1):
        train_loss = train(model, train_loader, optim, epoch_idx, file)
        train_losses.append(train_loss)

if __name__ == "__main__":
    if jt.has_cuda:
        jt.flags.use_cuda = 1
    parser = make_parser()
    main(parser.parse_args())