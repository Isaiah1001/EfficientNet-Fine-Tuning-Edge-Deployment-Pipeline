# preprocess/transforms_def.py

import torchvision.transforms as transforms
# data manipulate
def data_manipulate(mean, std, img_size=256):
    """apply data manipulation to the image

    Args:
        mean (tuple): mean for each channel
        std (tuple): std for each channel
    Returns:
        basic (torchvision.transforms.Compose): basic transform
        aug (torchvision.transforms.Compose): basic + augmented transform
    """
    transform_basic = transforms.Compose([
    transforms.Resize((img_size+20, img_size+20)),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
    ])
    
    transform_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomRotation(degrees=10)
        
    ])
    # augmentation for training data and basic transform for validation/test data
    basic = transforms.Compose([transform_basic])
    aug = transforms.Compose([transform_aug, transform_basic])
    return basic, aug