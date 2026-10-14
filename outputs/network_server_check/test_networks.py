"""Run over SSH stdin; do not modify the uploaded model files."""
import gc
import importlib.util
import json
import os
import sys
import time

sys.dont_write_bytecode = True
import torch
from torch import nn

torch.set_num_threads(2)
torch.manual_seed(20261014)
root = '/data/lym'

def emit(**data):
    print(json.dumps(data, ensure_ascii=False), flush=True)

def load(name):
    spec = importlib.util.spec_from_file_location('checked_' + name, root + '/' + name + '.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

emit(event='environment', python=sys.executable, torch=torch.__version__, device='cpu',
     cuda_available=torch.cuda.is_available(), threads=torch.get_num_threads())
resnet, cnn, efficientnet = [load(n) for n in ['resnet', 'cnn', 'efficientnet']]
factories = {
    'ResNet18': lambda: resnet.ResNet18(num_classes=11, in_channel=3),
    'ResNet50': lambda: resnet.ResNet50(num_classes=11, in_channel=3),
    'CNN': lambda: cnn.CNN(num_classes=11, in_channel=3),
    'EfficientNetB0': lambda: efficientnet.EfficientNetB0(num_classes=11, in_channels=3),
}
lengths = {'ResNet18': [128, 512, 3000], 'ResNet50': [128, 512],
           'CNN': [128, 512, 3000], 'EfficientNetB0': [128, 512, 3000]}
training_lengths = {'ResNet18': 512, 'ResNet50': 128, 'CNN': 3000, 'EfficientNetB0': 512}

for name, factory in factories.items():
    model = factory()
    emit(event='model', model=name, parameters=sum(p.numel() for p in model.parameters()),
         conv1d=sum(isinstance(m, nn.Conv1d) for m in model.modules()),
         conv2d=sum(isinstance(m, nn.Conv2d) for m in model.modules()))
    model.eval()
    for length in lengths[name]:
        started = time.monotonic()
        try:
            with torch.no_grad():
                output = model(torch.randn(2, 3, length))
            feature, logits = output if isinstance(output, tuple) else (None, output)
            assert logits.shape == (2, 11), tuple(logits.shape)
            assert torch.isfinite(logits).all()
            emit(event='forward', model=name, input=[2, 3, length], status='PASS',
                 logits=list(logits.shape), features=None if feature is None else list(feature.shape),
                 seconds=round(time.monotonic() - started, 3))
        except Exception as e:
            emit(event='forward', model=name, input=[2, 3, length], status='FAIL',
                 error_type=type(e).__name__, error=str(e))
    model.train()
    try:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        first = next(model.parameters())
        before = first.detach().clone()
        x = torch.randn(2, 3, training_lengths[name])
        labels = torch.tensor([0, 10])
        output = model(x)
        logits = output[1] if isinstance(output, tuple) else output
        loss = nn.CrossEntropyLoss()(logits, labels)
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        optimizer.step()
        changed = not torch.equal(before, first.detach())
        assert changed
        emit(event='train_step', model=name, input=list(x.shape), status='PASS',
             loss=round(loss.item(), 6), finite_gradients=True, first_parameter_updated=changed)
        del optimizer, first, before, x, labels, output, logits, loss, grads
    except Exception as e:
        emit(event='train_step', model=name, status='FAIL', error_type=type(e).__name__, error=str(e))
    if name == 'EfficientNetB0':
        emit(event='regularization', model=name,
             configured_drop_connect=model.cfg['drop_connect_rate'],
             actual_block_drop_rates=sorted({m.drop_rate for m in model.layers}),
             configured_dropout=model.cfg['dropout_rate'])
    del model
    gc.collect()

def cnn_final_length(length):
    for kernel in [3, 5, 7, 9, 11]:
        length = (length - 2 * (kernel - 1)) // 2
    return length

valid = [l for l in range(1, 4097) if cnn_final_length(l) == 19712 // 256]
emit(event='cnn_length_constraint', fc_inputs=19712, final_channels=256,
     compatible_length_min=min(valid), compatible_length_max=max(valid))
emit(event='done')
