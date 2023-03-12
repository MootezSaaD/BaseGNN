import copy
from sys import stderr

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm

from utils.utils import debug
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm

def evaluate_loss(model, loss_function, num_batches, data_iter, cuda=False):
    model.eval()
    with torch.no_grad():
        _loss = []
        all_predictions, all_targets = [], []
        for _ in range(num_batches):
            graph, targets = data_iter()
            targets = targets.cuda()
            predictions = model(graph, cuda=True)
            batch_loss = loss_function(predictions, targets)
            _loss.append(batch_loss.detach().cpu().item())
            predictions = predictions.detach().cpu()
            if predictions.ndim == 2:
                all_predictions.extend(np.argmax(predictions.numpy(), axis=-1).tolist())
            else:
                all_predictions.extend(
                    predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(
                        dtype=torch.int32).numpy().tolist()
                )
            all_targets.extend(targets.detach().cpu().numpy().tolist())
        model.train()
        return np.mean(_loss).item(), accuracy_score(all_targets, all_predictions) * 100
    pass


def evaluate_metrics(model, loss_function, num_batches, data_iter):
    model.eval()
    with torch.no_grad():
        _loss = []
        all_predictions, all_targets = [], []
        for _ in range(num_batches):
            graph, targets = data_iter()
            targets = targets.cuda()
            predictions = model(graph, cuda=True)
            batch_loss = loss_function(predictions, targets)
            _loss.append(batch_loss.detach().cpu().item())
            predictions = predictions.detach().cpu()
            if predictions.ndim == 2:
                all_predictions.extend(np.argmax(predictions.numpy(), axis=-1).tolist())
            else:
                all_predictions.extend(
                    predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(
                        dtype=torch.int32).numpy().tolist()
                )
            all_targets.extend(targets.detach().cpu().numpy().tolist())
        model.train()
        return accuracy_score(all_targets, all_predictions) * 100, \
               precision_score(all_targets, all_predictions) * 100, \
               recall_score(all_targets, all_predictions) * 100, \
               f1_score(all_targets, all_predictions) * 100
    pass

def measure_performance(model, device, test_loader, test_set, loss_fn):
    model.eval()
    with torch.no_grad():
        all_correct = 0
        epoch_size = 0
        losses = []
        for batch_data in test_loader:
            batch, label = batch_data
            batch = batch.to(device)
            label = label.float().unsqueeze(1).to(device)
            out = model(batch)
            loss = loss_fn(out, label)
            losses.append(loss.item() * len(out))
            pred = torch.gt(out, 0.5)
            correct = pred == label
            all_correct += correct.sum().item()
            epoch_size += len(label)
        acc = all_correct / epoch_size
        return acc, sum(losses) / len(test_set)
    
def train(model, train_set, val_set, batch_size, max_steps, dev_every, loss_function, optimizer, save_path, log_every=50, max_patience=5, device=None):
    debug('Start Training')
    train_losses = []
    best_model = None
    patience_counter = 0
    best_f1 = 0
    train_loader = GraphDataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = GraphDataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False
    )
    try:
     for i in range(500):
        epoch_loss = 0
        # torch_geometric.loader.DataLoader concatenates all the graphs in the batch
        # into one big disjoint graph, so we can train with a batch as if it's a single graph.
        model.train()
        pbar = tqdm(total=len(train_set))
        for j, batch_data in enumerate(train_loader):
            pbar.set_postfix({"epoch": i, "batch": j})
            batch, label = batch_data
            label = label.float().unsqueeze(1).to(device) # go from size([batch_size]) to size([batch_size, 1])
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_function(out, label)
            loss.backward()
            epoch_loss += loss.item() * batch.batch_size
            optimizer.step()
            pbar.update(batch.batch_size)

        train_loss = epoch_loss / len(train_set)
        train_acc, _ = measure_performance(model, device, train_loader, train_set, loss_function)
        test_acc, test_loss = measure_performance(model, device, val_loader, train_set, loss_function)
        if i % log_every == 0:
            print('Epoch:', i, 'Train loss:', train_loss, 'Test Accuracy:', test_acc)
        # scheduler.step()
    except KeyboardInterrupt:
        debug('Training Interrupted by user!')

    # if best_model is not None:
    #     model.load_state_dict(best_model)
    # _save_file = open(save_path + '-model.bin', 'wb')
    # torch.save(model.state_dict(), _save_file)
    # _save_file.close()
    # acc, pr, rc, f1 = evaluate_metrics(model, loss_function, dataset.initialize_train_batch(),
    #                                    dataset.get_next_train_batch)
    # debug('%s\tTest Accuracy: %0.2f\tPrecision: %0.2f\tRecall: %0.2f\tF1: %0.2f' % (save_path, acc, pr, rc, f1))
    # debug('=' * 100)
