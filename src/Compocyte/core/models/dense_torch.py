import numpy as np
import os
import pickle
import logging
logger = logging.getLogger(__name__)

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    _TORCH_AVAILABLE = False


def _require_torch():
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "DenseTorch requires the 'torch' extra: "
            "pip install \"Compocyte[torch]\"")
    return torch

try:
    from scipy import sparse as _sparse
except ImportError:  # pragma: no cover - scipy is a hard dependency of Compocyte
    _sparse = None


def resolve_device(device=None):
    """Resolve the torch device to run inference/training on.

    Explicit ``device`` wins; otherwise CUDA is preferred over MPS over CPU.
    Always returns a ``torch.device``; CPU-only machines transparently fall
    back to ``'cpu'`` so all call sites stay device-agnostic.
    """
    _torch = _require_torch()
    if device is not None:
        return _torch.device(device) if not isinstance(device, _torch.device) else device
    if _torch.cuda.is_available():
        return _torch.device('cuda')
    if hasattr(_torch.backends, 'mps') and _torch.backends.mps.is_available():
        return _torch.device('mps')
    return _torch.device('cpu')


def _to_dense_float32(x):
    if _sparse is not None and _sparse.issparse(x):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float32)
    return x

if _TORCH_AVAILABLE:
    _DenseTorchBase = torch.nn.Module
else:  # torch-free installs: importing the module must still work so that
    # isinstance checks and the classifier registry keep functioning. Only
    # instantiating or calling raises, with a pointer to the right extra.
    class _DenseTorchBase:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            _require_torch()


class DenseTorch(_DenseTorchBase):
    def __init__(
            self, 
            labels: list, 
            n_input: int, 
            n_output: int,
            hidden_layers: list=[64, 64],
            dropout: float=0.4,
            batchnorm: bool=True,
            device=None):
        
        super().__init__()

        self.device = str(resolve_device(device))

        self.labels_enc = {label: i for i, label in enumerate(labels)}
        self.labels_dec = {self.labels_enc[label]: label for label in self.labels_enc.keys()}
        self.layers = torch.nn.ModuleList()
        layers = [n_input] + hidden_layers + [n_output]
        for i in range(len(layers) - 1):
            n_in = layers[i]
            n_out = layers[i + 1]
            new_linear = torch.nn.Linear(n_in, n_out)
            new_activation = torch.nn.LeakyReLU(0.1)
            new_batchnorm = torch.nn.BatchNorm1d(n_out)
            new_dropout = torch.nn.Dropout(dropout)
            torch.nn.init.xavier_uniform_(
                new_linear.weight, 
                gain=torch.nn.init.calculate_gain('leaky_relu', 0.1))
            torch.nn.init.zeros_(new_linear.bias)
            self.layers.append(new_linear)
            if i < (len(layers) - 2):
                self.layers.append(new_activation)
                if batchnorm: 
                    self.layers.append(new_batchnorm)

                self.layers.append(new_dropout)

            else:
                self.layers.append(
                    torch.nn.Softmax(dim=1)
                )

    def forward(self, x):
        for layer in self.layers:            
            x = layer(x)

        return x

    def get_device(self):
        return resolve_device(getattr(self, 'device', None))

    def to_device(self, device=None):
        self.device = str(resolve_device(device))
        return super().to(self.device)

    def predict_logits(self, x, batch_size=8192, device=None) -> np.array:
        dev = resolve_device(device or getattr(self, 'device', None))
        self.to(dev)
        self.eval()
        x = _to_dense_float32(x)
        n_output = len(self.labels_dec)
        out = np.empty((x.shape[0], n_output), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, x.shape[0], batch_size):
                xb = torch.from_numpy(x[start:start + batch_size]).to(dev)
                out[start:start + xb.shape[0]] = self(xb).detach().to('cpu').numpy()

        return out

    def predict_logits_mc(
            self, x, monte_carlo: int,
            dropout_p: float=0.5,
            batch_size: int=4096,
            mc_max_rows: int=65536,
            device=None) -> np.array:
        """Vectorized Monte Carlo input-feature dropout.

        Preserves the historical semantics (input masking with dropout scaling,
        internal dropout off, BatchNorm frozen in eval) while replacing the
        per-iteration NumPy<->torch round-trips with one fused forward pass per
        cell chunk: inputs are repeated ``monte_carlo`` times along the batch
        axis, so ``BatchNorm1d`` (eval mode) and ``Linear`` stay mathematically
        identical per row. Returns ``(monte_carlo, n_cells, n_classes)``.
        """
        dev = resolve_device(device or getattr(self, 'device', None))
        self.to(dev)
        self.eval()  # keep BatchNorm frozen and internal dropout off by design
        x = _to_dense_float32(x)
        n_cells, n_features = x.shape
        n_output = len(self.labels_dec)
        scale = 1.0 / (1.0 - dropout_p)
        cell_chunk = max(1, min(batch_size, mc_max_rows // max(1, monte_carlo)))
        all_logits = np.empty((monte_carlo, n_cells, n_output), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, n_cells, cell_chunk):
                end = min(start + cell_chunk, n_cells)
                base = torch.from_numpy(x[start:end]).to(dev)
                rows = monte_carlo * (end - start)
                repeated = base.repeat(monte_carlo, 1)
                mask = (torch.rand((rows, n_features), device=dev) >= dropout_p).to(torch.float32)
                logits = self(repeated * (mask * scale)).detach().to('cpu').numpy()
                all_logits[:, start:end, :] = logits.reshape(monte_carlo, end - start, n_output)

        return all_logits

    def predict(self, x, batch_size=8192, device=None) -> np.array:        
        logits = self.predict_logits(x, batch_size=batch_size, device=device)
        pred = np.argmax(logits, axis=1)
        pred = np.array(
            [self.labels_dec[p] for p in pred]
        )

        return pred    

    def reset_output(self, n_output):
        in_features = self.layers[-2].in_features
        del self.layers[-2] # last dense
        del self.layers[-1] # softmax
        self.layers.append(
            torch.nn.Linear(in_features, n_output)
        )
        torch.nn.init.xavier_uniform_(self.layers[-1].weight, gain=torch.nn.init.calculate_gain('leaky_relu', 0.1))
        torch.nn.init.zeros_(self.layers[-1].bias)
        self.layers.append(
            torch.nn.Softmax(dim=1)
        )

    def state_to_dict(self):
        """Serialize model weights and architecture to plain Python/numpy objects.

        Avoids returning a live ``torch.nn.Module`` across a ``multiprocessing.Pool``,
        which would otherwise rely on PyTorch's file-descriptor tensor sharing and
        can deadlock when workers exit before the parent unpickles the FDs.
        """
        n_input = self.layers[0].in_features
        n_output = self.layers[-2].out_features
        hidden_layers = []
        dropout = 0.0
        batchnorm = False
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                hidden_layers.append(layer.out_features)
            elif isinstance(layer, torch.nn.Dropout):
                dropout = layer.p
            elif isinstance(layer, torch.nn.BatchNorm1d):
                batchnorm = True
        # Last linear output is n_output, so intermediate hidden layers are all
        # Linear outputs except the final one.
        hidden_layers = hidden_layers[:-1]
        return {
            '__dense_torch_state__': True,
            'state_dict': {k: v.detach().cpu().numpy() for k, v in self.state_dict().items()},
            'labels_enc': self.labels_enc,
            'labels_dec': self.labels_dec,
            'n_input': n_input,
            'n_output': n_output,
            'hidden_layers': hidden_layers,
            'dropout': dropout,
            'batchnorm': batchnorm,
            'device': str(self.device),
        }

    @classmethod
    def from_state_dict(cls, d):
        """Reconstruct a DenseTorch from a dict produced by :meth:`state_to_dict`."""
        model = cls(
            labels=list(d['labels_enc'].keys()),
            n_input=d['n_input'],
            n_output=d['n_output'],
            hidden_layers=d['hidden_layers'],
            dropout=d.get('dropout', 0.0),
            batchnorm=d.get('batchnorm', True),
            device=d.get('device', 'cpu'),
        )
        state = {k: torch.from_numpy(v) for k, v in d['state_dict'].items()}
        model.load_state_dict(state)
        model.is_fitted = True
        return model

    def _save(self, path):
        non_param_attr = ['histories', 'labels_enc', 'labels_dec', 'device']
        non_param_dict = {}
        for item in self.__dict__.keys():
            if item in non_param_attr:
                non_param_dict[item] = self.__dict__[item]

        torch.save(self, os.path.join(path, 'model'))
        with open(os.path.join(path, 'non_param_dict.pickle'), 'wb') as f:
            pickle.dump(non_param_dict, f)

    @classmethod
    def _load(cls, path):
        _torch = _require_torch()
        model = _torch.load(os.path.join(path, 'model'), map_location='cpu', weights_only=False)
        with open(os.path.join(path, 'non_param_dict.pickle'), 'rb') as f:
            non_param_dict = pickle.load(f)

        for item in non_param_dict.keys():
            model.__dict__[item] = non_param_dict[item]

        if not hasattr(model, 'device'):
            model.device = 'cpu'

        return model

    @classmethod
    def import_external(
        cls,
        model,
        labels,):

        if not issubclass(type(model), torch.nn.Module):
            raise TypeError('To import an external model as DenseTorch, it must be a subclass of torch.nn.Module.')

        denseTorch = cls(labels, 2, 2)
        denseTorch.layers = torch.nn.ModuleList([model])

        return denseTorch