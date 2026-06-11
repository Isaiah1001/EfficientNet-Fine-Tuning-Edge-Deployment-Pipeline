from torch.utils.data import Subset

# subset class to apply different transforms
class subsetTrans:
    """apply different transforms to different subsets of the dataset
    """
    def __init__(self, subset: Subset, transform = None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label, description = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, description
