# preprocess/data_split_subset.py
from torch.utils.data import Subset

def get_dataset(dataset,split_data):
    """Split dataset into three subsets, training, validaion and test

    Args:
        dataset (tuple): loaded dataset
        split_data (array): splited ID groups

    Returns:
        subdatasets: training, valication and test subsets
    """
    
    train_ids = split_data["trnid"][0] - 1
    val_ids   = split_data["valid"][0] - 1
    test_ids  = split_data["tstid"][0] - 1

    print(f"Train samples: {len(train_ids)}") 
    print(f"Val samples:   {len(val_ids)}")
    print(f"Test samples:  {len(test_ids)}")

    # create subsets
    train_dataset = Subset(dataset, train_ids.tolist())
    val_dataset   = Subset(dataset, val_ids.tolist())
    test_dataset  = Subset(dataset, test_ids.tolist())

    return train_dataset, val_dataset, test_dataset 