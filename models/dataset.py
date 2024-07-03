import torch.utils.data as data
class LCDataset(data.Dataset):

    def __init__(self, data, label):
        """
        Inputs:
            size - Number of data points we want to generate
            std - Standard deviation of the noise (see generate_continuous_xor function)
        """
        super().__init__()
        self.data = data
        self.label = label

    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.data.size(0)

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        data_point = self.data[idx]
        data_label = self.label[idx]
        return data_point, data_label