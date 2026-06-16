# ./base_flower.py
# =========================
# 0) Imports
# =========================
# libraries
import os
import torch
import torchvision.models as tv_models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import mlflow

import lightning.pytorch as pl
# from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import LearningRateMonitor, BaseFinetuning, ModelCheckpoint
# from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.cli import LightningCLI, OptimizerCallable, LRSchedulerCallable
from scipy.io import loadmat

from preprocess import data_access, get_dataset, data_manipulate, subsetTrans
torch.set_float32_matmul_precision('medium') 

# ==============================================
# 1) load pre-trained weights and get the default transforms
# ==============================================
weights = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1
auto_transforms = weights.transforms()
model_input_size = auto_transforms.crop_size[0] 


# ==============================================
# 2) lightning data module setup
# ==============================================

class FlowerDataModule(pl.LightningDataModule):
    def __init__(self,
                 bs: int = 128, 
                 nw: int = 2,
                 pf: int = 2,
                 data_path: str = "./99_flower_data",
                 splitid: str = "./99_flower_data/setid.mat",
                 image_size = model_input_size
                 ):
        """initialize the data module
        Args:
            data_path: path to the dataset
            train_transform: transforms to apply to the training data
            basic_transform: transforms to apply to the validation and test data
            batch_size (int, optional): batch size. Defaults to 32.
            num_workers (int, optional): number of workers for data loading. Defaults to 0.
        """
        super().__init__()
        self.save_hyperparameters()
        
        # normalize dataset
        mean=[0.485, 0.456, 0.406] 
        std=[0.229, 0.224, 0.225] # using the mean and std from ImageNet dataset
        self.basic_transform, self.train_transform = data_manipulate(mean, std, img_size=self.hparams.image_size)
        
        
    def setup(self, stage=None):
        """setup the data module by applying transforms and creating datasets
        """
        # read data using predefined function
        dataset = data_access(self.hparams.data_path)
        split_data = loadmat(self.hparams.splitid) 
        self.train_dataset, self.val_dataset, self.test_dataset = get_dataset(dataset,split_data)

    def train_dataloader(self):
        """return training data loader
        """
        return DataLoader(self.train_dataset, batch_size=self.hparams.bs, num_workers=self.hparams.nw, 
                          shuffle=True, pin_memory=False, prefetch_factor=self.hparams.pf, persistent_workers=True)

    def val_dataloader(self):
        """return validation data loader
        """
        return DataLoader(self.val_dataset, batch_size=self.hparams.bs, num_workers=self.hparams.nw, 
                          persistent_workers=True, pin_memory=False )

    def test_dataloader(self):
        """return test data loader
        """
        return DataLoader(self.test_dataset, batch_size=self.hparams.bs, num_workers=self.hparams.nw, 
                          persistent_workers=True, pin_memory=False)

# ==============================================
# 3) lightning module setup
# ==============================================

class FlowerLightModule(pl.LightningModule):
    def __init__(self, num_classes: int=102,
                 optimizer: OptimizerCallable = torch.optim.SGD,
                 scheduler: LRSchedulerCallable = torch.optim.lr_scheduler.ConstantLR):
        """initialize the model module

        Args:
            model: the neural network model
            learning_rate (float, optional): learning rate. Defaults to 0.001.
            weight_decay (float, optional): weight decay for optimizer. Defaults to 0.0001.
            device (str, optional): device to run the model on. Defaults to None (will use 'cuda' if available).
        """
        super().__init__()
        # Save the hyperparameters passed to the constructor. This makes them
        # accessible via `self.hparams` and logs them automatically.
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.lr_scheduler = scheduler
        # Import the pre-trained EfficientNet-B0 model and modify the classifier head.
        self.model = tv_models.efficientnet_b0(weights='IMAGENET1K_V1')
        in_features = self.model.classifier[1].in_features
        new_fc_layer = torch.nn.Linear(in_features, self.hparams.num_classes)
        self.model.classifier[1] = new_fc_layer
        # Initialize the loss function.
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        # Initialize metrics to track accuracy for training and validation.
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x: The input tensor containing a batch of images.

        Returns:
            The output tensor (logits) from the model.
        """
        x = self.model(x)
        return x
    
    def training_step(self, batch, batch_idx = None):
        # if batch_idx == 0:
        #     print(f"[training_step] self.training       = {self.training}")
        #     print(f"[training_step] self.model.training = {self.model.training}")
        inputs, labels, _ = batch
        outputs = self(inputs)
        # update and log training loss 
        loss = self.loss_fn(outputs, labels)
        bs = inputs.size(0)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        # update and log training accuracy
        self.train_accuracy.update(outputs, labels)
        self.log("train_acc", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        
        return loss
    
    def validation_step(self, batch, batch_idx=None):
        inputs, labels, _ = batch
        outputs = self(inputs)
        # update and log validation loss
        loss = self.loss_fn(outputs, labels)
        bs = inputs.size(0)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
        # update and log validation accuracy
        self.val_accuracy.update(outputs, labels)
        self.log("val_acc", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True, batch_size=bs)
    
    def on_train_epoch_end(self):
        pass
        # return self.train_accuracy.reset()
    
    def on_validation_epoch_end(self):
        pass
        # return self.val_accuracy.reset()
    
    def configure_optimizers(self):
        optimizer = self.optimizer(
            filter(lambda p: p.requires_grad, self.parameters()))

        scheduler = self.lr_scheduler(optimizer)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=10, eta_min=1e-4
        # )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
        
# ==============================================
# 4) custom some callbacks
# ==============================================
# progressive fine-tuning callback setup
class ProgressiveBackboneFinetuning(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch_1=5, unfreeze_at_epoch_2=10):
        """define specific epoch to start and stop fine tuning

        Args:
            unfreeze_at_epoch_1 (int, optional): epoch to unfreeze the last three layers of the backbone. Defaults to 5.
            unfreeze_at_epoch_2 (int, optional): epoch to unfreeze the last five layers of the backbone. Defaults to 10.
        """
        super().__init__()
        self.unfreeze_at_epoch_1 = unfreeze_at_epoch_1
        self.unfreeze_at_epoch_2 = unfreeze_at_epoch_2

    def freeze_before_training(self, pl_module):
        """freeze all layers before training"""
        self.freeze(pl_module.model.features, train_bn=True) 
        """unfreeze the classifier head before training"""
        self.make_trainable(pl_module.model.classifier)

    def finetune_function(self, pl_module, epoch, optimizer):
        if (self.unfreeze_at_epoch_1 is not None and epoch == self.unfreeze_at_epoch_1):
            self.unfreeze_and_add_param_group(
                modules=pl_module.model.features[-1:],
                optimizer=optimizer,
                lr=optimizer.defaults['lr'] * 0.1
                
            )
            
        if (self.unfreeze_at_epoch_2 is not None and self.unfreeze_at_epoch_1 is not None and epoch == self.unfreeze_at_epoch_2):
            self.unfreeze_and_add_param_group(
                modules=pl_module.model.features[-3:-1],
                optimizer=optimizer,
                lr=optimizer.defaults['lr'] * 0.02
                # initial_denom_lr=10
            )
        if (self.unfreeze_at_epoch_2 is not None and self.unfreeze_at_epoch_1 is None and epoch == self.unfreeze_at_epoch_2):
            self.unfreeze_and_add_param_group(
                modules=pl_module.model.features[-3:],
                optimizer=optimizer,
                lr=optimizer.defaults['lr'] * 0.02
                # initial_denom_lr=10
            )

# model summary callback setup: prints whenever trainable params change
class PostFreezeModelSummary(pl.Callback):
    def __init__(self):
        super().__init__()
        self._last_trainable = None

    def _print_if_changed(self, pl_module, epoch=None):
        total = sum(p.numel() for p in pl_module.parameters())
        trainable = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        if trainable != self._last_trainable:
            frozen = total - trainable
            label = f"Epoch {epoch}" if epoch is not None else "Train Start"
            print(f"\n{'='*50}")
            print(f"  [{label}] Trainable params changed!")
            print(f"  Total params:     {total:,}")
            print(f"  Trainable params: {trainable:,}")
            print(f"  Frozen params:    {frozen:,}")
            print(f"  Trainable %:      {trainable/total*100:.1f}%")
            print(f"{'='*50}\n")
            self._last_trainable = trainable

    def on_train_start(self, trainer, pl_module):
        self._print_if_changed(pl_module)

    def on_train_epoch_start(self, trainer, pl_module):
        self._print_if_changed(pl_module, epoch=trainer.current_epoch)

# ==============================================
# 5) CLI setup
# ==============================================
def cli_main():
    LightningCLI(
        FlowerLightModule,
        FlowerDataModule,
        seed_everything_default=42,
        save_config_callback=None,
        save_config_kwargs=True,
        # auto_configure_optimizers=False
    )

if __name__ == "__main__":
    cli_main()