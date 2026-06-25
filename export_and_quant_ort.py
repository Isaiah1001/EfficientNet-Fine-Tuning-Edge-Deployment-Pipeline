import numpy as np
import torch

from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    CalibrationMethod,
)

from base_flower import FlowerLightModule, FlowerDataModule


CKPT_PATH = "./logs/checkpoints/checkpoint_base_epoch=36_val_acc=0.9225.ckpt"
DATA_PATH = "./99_flower_data"

FP32_ONNX = "flower_efficientnet_basemodel.onnx"
INT8_ONNX = "flower_efficientnet_quantization.onnx"


class FlowerDataReader(CalibrationDataReader):
    def __init__(self, dataloader, input_name="input", max_samples=300):
        self.input_name = input_name
        self.samples = []

        count = 0
        for batch in dataloader:
            images, labels, decription = batch
            images = images.cpu().numpy().astype(np.float32)

            for j in range(images.shape[0]):
                self.samples.append({self.input_name: images[j:j+1]})
                count += 1
                if max_samples is not None and count >= max_samples:
                    self.iterator = iter(self.samples)
                    return

        self.iterator = iter(self.samples)

    def get_next(self):
        return next(self.iterator, None)


def export_onnx():
    lit_model = FlowerLightModule.load_from_checkpoint(CKPT_PATH, map_location="cpu", weights_only=False,)
    model = lit_model.model
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy,
        FP32_ONNX,
        input_names=["input"],
        output_names=["logits"],
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )

    print(f"Exported FP32 ONNX: {FP32_ONNX}")


def quantize_onnx():
    dm = FlowerDataModule(
        bs=1,
        nw=2,
        data_path=DATA_PATH,
        image_size=224,
    )
    dm.setup("fit")
    val_loader = dm.val_dataloader()

    reader = FlowerDataReader(
        val_loader,
        input_name="input",
        max_samples=300,
    )

    quantize_static(
        model_input=FP32_ONNX,
        model_output=INT8_ONNX,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.Percentile,
    )

    print(f"Quantized INT8 ONNX: {INT8_ONNX}")


if __name__ == "__main__":
    export_onnx()
    quantize_onnx()
    print("Done:", INT8_ONNX)
    