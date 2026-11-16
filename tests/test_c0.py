"""C0 冻结策略和验证指标的回归检查。"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.c0 import C0ResNet18
from src.models.c0_training import evaluate


def test_only_fc_trains_and_batchnorm_stats_stay_fixed():
    torch.manual_seed(42)
    model = C0ResNet18(pretrained=False)
    model.train()
    assert all(not layer.training for layer in model.modules() if isinstance(layer, nn.BatchNorm2d))
    assert {name for name, parameter in model.named_parameters() if parameter.requires_grad} == {
        "network.fc.weight", "network.fc.bias"
    }
    before = {name: value.clone() for name, value in model.named_buffers() if "running_" in name}
    images = torch.randn(2, 3, 64, 64)
    labels = torch.tensor([0, 5])
    optimizer = torch.optim.Adam(model.trainable_parameters, lr=0.001)
    old_fc = model.network.fc.weight.detach().clone()
    loss = nn.CrossEntropyLoss()(model(images), labels)
    loss.backward()
    optimizer.step()
    assert model.network.fc.weight.grad is not None
    assert not torch.equal(old_fc, model.network.fc.weight)
    assert all(parameter.grad is None for name, parameter in model.named_parameters() if not name.startswith("network.fc."))
    assert all(torch.equal(value, dict(model.named_buffers())[name]) for name, value in before.items())


def test_validation_metrics_include_all_six_classes():
    class FixedModel(nn.Module):
        def forward(self, images):
            return images

    scores = torch.tensor([[3., 0., 0., 0., 0., 0.], [3., 0., 0., 0., 0., 0.]])
    labels = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(scores, labels), batch_size=2)
    metrics, logits, actual = evaluate(FixedModel(), loader, nn.CrossEntropyLoss(), torch.device("cpu"))
    assert logits.shape == (2, 6)
    assert torch.equal(actual, labels)
    assert metrics["accuracy"] == 0.5
    assert abs(metrics["macro_f1"] - (2 / 3) / 6) < 1e-8
