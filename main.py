import argparse
import os
import sys

import numpy as np
from data_loader.lit_data import DataModule
import torch
from torch.nn import BCELoss
import pytorch_lightning as pl

from modules.ggnn import GGNN
from utils.data import static_splitter
from modules.lit_classifier import PlastDectClassifier
from utils.init_features import load_wv


if __name__ == '__main__':
    torch.manual_seed(1000)
    np.random.seed(1000)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, help='Type of the model (devign/ggnn)',
                        choices=['devign', 'ggnn'], default='ggnn')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset for experiment.')

    parser.add_argument('--data_src', type=str, help='CSV file of the dataset.', required=True)
    parser.add_argument('--feature_size', type=int, help='Size of feature vector for each node', default=128)
    parser.add_argument('--graph_embed_size', type=int, help='Size of the Graph Embedding', default=200)
    parser.add_argument('--num_steps', type=int, help='Number of steps in GGNN', default=6)
    parser.add_argument('--batch_size', type=int, help='Batch Size for training', default=32)
    parser.add_argument('--read_out', type=str, help='GNN readout function', choices=['sum', 'mean'], default='sum')
    parser.add_argument('--emb_type', type=str, help='Embedding method for node feature generation. Wor2Vec or Transformer-based', choices=['w2v', 'hf'], default='w2v')
    parser.add_argument('--w2v', type=str, help='Pretrained Word2Vec model path, when w2v is selected as an embedding method.')
    parser.add_argument('--tok', type=str, help='Tokenization method of the src code.', choices=['nltk', 'hf'], default='nltk')
    parser.add_argument('--build_method', type=str, help='Graph construction method. UTC or IFC.', choices=['utc', 'ifc'], default='ifc')
    parser.add_argument('--window_size', type=int, help='Sliding window size.', default=3)
    args = parser.parse_args()

    if args.feature_size > args.graph_embed_size:
        print('Warning!!! Graph Embed dimension should be at least equal to the feature dimension.\n'
              'Setting graph embedding size to feature size', file=sys.stderr)
        args.graph_embed_size = args.feature_size

    model_dir = os.path.join('models', args.dataset)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    train_split, test_split, val_split =  static_splitter(args.data_src)

    # If we are using Word2Vec, load it just once
    if args.emb_type == 'w2v':
        args.w2v_model = load_wv(args.w2v)
    
    data_module = DataModule(args.data_src, args)
    

    graph_model = GGNN(input_dim=args.feature_size, output_dim=args.graph_embed_size,
                        num_steps=args.num_steps, max_edge_types=1, read_out=args.read_out)
    
    loss_function = torch.nn.BCELoss(reduction='sum')
    model = PlastDectClassifier(graph_model, loss_function)
    early_stopping_cb = pl.callbacks.early_stopping.EarlyStopping(monitor="val_f1", mode="max", patience=5)
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[0],
        progress_bar_refresh_rate=20,
        max_epochs=100,
        callbacks=[early_stopping_cb]
    )
    
