from dgl.data import DGLDataset
import torch
import pandas as pd
from utils.graph_construction import utc, ifc
from utils.init_features import load_wv, nltk_tokenizer
from utils.io import read_file
from utils.utils import debug
from data_loader.data_entry import DataEntry

class PlaseDectDataset(DGLDataset):
    """ Template for customizing graph datasets in DGL.
    Parameters
    ----------
    url : str
        URL to download the raw dataset
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    save_dir : str
        Directory to save the processed dataset.
        Default: the value of `raw_dir`
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information
    """
    def __init__(self,
                 df,
                 args,
                 url=None,
                 raw_dir=None,
                 save_dir=None,
                 force_reload=False,
                 verbose=False):
        self.args = args
        self.labels = df['label'].tolist()
        self.graphs = df['file'].tolist()
        self.split_name = df['split'].iloc[0]
        print(f"Number of {self.split_name} instances: {len(self.labels)}")
        self.emb_type = args.emb_type
        if self.emb_type == 'w2v':
            self.embeddings = args.w2v_model
        self.build_method = args.build_method
        super(PlaseDectDataset, self).__init__(name=f'plasdect_{self.split_name}',
                                           url=url,
                                           raw_dir=raw_dir,
                                           save_dir=save_dir,
                                           force_reload=force_reload,
                                           verbose=verbose)

    def download(self):
        # download raw data to local disk
        pass

    def process(self):
        pass

    def __getitem__(self, idx):
        # get one example by index
        src_code, label = read_file(self.graphs[idx]), self.labels[idx]
        if self.emb_type == 'w2v':
            if self.args.tok == 'nltk':
                tokens = nltk_tokenizer(src_code)
            if self.args.build_method == 'ifc':
                adj, features = ifc(tokens, self.embeddings, self.args)
            if self.args.build_method == 'utc':
                adj, features = utc(tokens, self.embeddings, self.args)
            
            sample: DataEntry = DataEntry(self.split_name, adj, features, label)

        return sample.graph, label

    def __len__(self):
        # number of data examples
        return len(self.graphs)

    @property
    def processed_file(self):
        pass

    @property
    def max_etypes(self):
        return 1

    def save(self):
        # save processed data to directory `self.save_path`
        pass

    def load(self):
        # load processed data from directory `self.save_path`
        pass

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        pass
