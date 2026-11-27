"""C0 初始化的部分微调，所有 BatchNorm 运行统计量固定。"""

from src.models.c0 import C0ResNet18


class C1ResNet18(C0ResNet18):
    def __init__(self):
        super().__init__(pretrained=False)
        for name, parameter in self.network.named_parameters():
            parameter.requires_grad_(name.startswith(("layer4.", "fc.")))

    @property
    def trainable_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]
