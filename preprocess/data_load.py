import os
from scipy.io import loadmat
from PIL import Image

class data_access:
    """
    Access data and prepare the data for next step in the pipeline.
    """
    def __init__(self, data_folder):
        # data path setup
        self.data_path = os.path.join('../', data_folder)
        self.image_folder = os.path.join(self.data_path, 'jpg')
        # load all labels, minus 1 to make it zero-indexed
        self.data_labels = loadmat(os.path.join(self.data_path, 'imagelabels.mat'))['labels'][0]-1
    # define methods to retrieve image
    def retrieve_images(self, idx):
        """load image data to 

        Args:
            idx (int): index of the image to load
        
        Returns:
            image (PIL.Image): loaded image
        """
        
        # load image, file name format: image_00001.jpg, ...
        image_path = os.path.join(self.image_folder, f'image_{1 + idx:05d}.jpg')
        with Image.open(image_path) as image:
            # convert to RGB
            image = image.convert('RGB')
        return image
    
    def retrieve_labels(self, index):
        """load label data
        
        Args:
            index (int): index of the label to load
    
        Returns:
            label (int): loaded label
        """
        
        data_label = loadmat(os.path.join(self.data_path, 'imagelabels.mat'))
        label = data_label['labels'][0, index].item()-1
        return label
    
    def retrieve_description(self, label):
        """load description data
        
        Args:
            index (int): index of the description to load
    
        Returns:
            description (str): loaded description
        """
        
        with open(os.path.join(self.data_path, "labels_description.txt"), "r", encoding="utf-8") as f:
            descriptions = f.readlines()
        description = descriptions[label].strip()
        return description
    
    def __len__(self):
        """return the total number of images in the dataset"""
        return len(self.data_labels)
    
    def __getitem__(self, index):
        """get image, label, and description by index
        
        Args:
            index (int): index of the data to retrieve
        
        Returns:
            dict: dictionary containing image, label, and description
        """
        image = self.retrieve_images(index)
        label = self.retrieve_labels(index)
        description = self.retrieve_description(label)
        return image, label, description
    
    