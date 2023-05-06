import pandas as pd  
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class ERDataset(Dataset):
    def __init__(self, data, position, train, test_fraction, random_seed):
        # positions as timepoints here
        train_data, test_data, train_pos, test_pos = train_test_split(
            data, position, test_size=test_fraction, random_state=random_seed)
        
        self.train_pos, self.test_pos = train_pos, test_pos
        self.train_data, self.test_data = train_data, test_data
        self.data = self.train_data if train else self.test_data
        self.pos = self.train_pos if train else self.test_pos

    def __getitem__(self, index):
        return self.data[index], self.pos[index]

    def __len__(self):
        return len(self.data)

class CD4(ERDataset):
    def __init__(self, train=True, test_fraction=0.1, seed=42):
        #here pos are actually class labels, just conforming with parent class!
        data = pd.read_csv('/ix/djishnu/Hanxi/23_ICML/topological-autoencoders/data/CD4_Data/w6_single.csv', index_col=0).values
        labels = np.ones(data.shape[0]) #dummy labels
        pos = labels
        data = data.astype(np.float32)
        pos = pos.astype(np.float32)
        _rnd = np.random.RandomState(seed)
        super().__init__(data, pos, train, test_fraction, _rnd)
