from copy import deepcopy
import os
from typing import Union
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import robust_scale
import torch
import logging
import dask.array as da
from torch.utils.data import Dataset, TensorDataset, random_split, DataLoader, IterableDataset
from Compocyte.core.models.dense_torch import DenseTorch
from Compocyte.core.models.dummy_classifier import DummyClassifier
from Compocyte.core.models.log_reg import LogisticRegression
from Compocyte.core.models.trees import BoostedTrees
from keras.utils import to_categorical
from balanced_loss import Loss as BalancedLoss

logger = logging.getLogger(__name__)


class DaskBatchDataset(IterableDataset):
    def __init__(self, X, y):
        # Convert both to lists of delayed chunks
        self.X_chunks = X.to_delayed().ravel()
        self.y_chunks = y.to_delayed().ravel()

        assert len(self.X_chunks) == len(self.y_chunks), \
            "Feature and label chunks must be aligned"

    def __iter__(self):
        for X_chunk, y_chunk in zip(self.X_chunks, self.y_chunks):
            X_np = X_chunk.compute()
            y_np = y_chunk.compute()
            yield (
                torch.from_numpy(X_np).to(torch.float32),
                torch.from_numpy(y_np).to(torch.float32)
            )

def predict_logits(model, x):
    x = robust_scale(x, axis=1, with_centering=False, copy=False, unit_variance=True)
    if hasattr(x, 'todense'):
        x = x.todense()
        
    if isinstance(model, DenseTorch):
        logits = model.predict_logits(x)
    
    elif isinstance(model, LogisticRegression):
        logits = model.predict_logits(x)

    elif isinstance(model, BoostedTrees):
        logits = model.predict_logits(x)

    elif isinstance(model, DummyClassifier):
        logits = model.predict_logits(x)

    else:
        raise Exception('Unknown classifier type.')

    return logits

def predict(model, x, threshold=-1, monte_carlo: int=None):
    if monte_carlo is not None:
        all_logits = []
        dropout = torch.nn.Dropout(p=0.5)
        for _ in range(monte_carlo):            
            x_dropout = np.array(dropout(torch.Tensor(x)))
            if isinstance(model, DenseTorch):
                all_logits.append(model.predict_logits(x_dropout))
            
            elif isinstance(model, LogisticRegression):
                all_logits.append(model.predict_logits(x_dropout))

            elif isinstance(model, BoostedTrees):
                all_logits.append(model.predict_logits(x_dropout))

            elif isinstance(model, DummyClassifier):
                return model.predict(x)
            
            else:
                raise Exception('Unknown classifier type')
            
        all_logits = np.array(all_logits)
        logits = np.mean(all_logits, axis=0)
    
    else:
        if isinstance(model, DenseTorch):
            logits = model.predict_logits(x)
        
        elif isinstance(model, LogisticRegression):
            logits = model.predict_logits(x)

        elif isinstance(model, BoostedTrees):
            logits = model.predict_logits(x)

        elif isinstance(model, DummyClassifier):
            return model.predict(x)
        
        else:
            raise Exception('Unknown classifier type')
        
    max_activation = np.max(logits, axis=1)
    pred = np.argmax(logits, axis=1).astype(int)
    pred = np.array([model.labels_dec[p] for p in pred])
    pred[max_activation <= threshold] = ''

    if monte_carlo is not None:
        return pred, all_logits

    return pred

def samples_per_class(y):
    spc = list(torch.zeros(y.shape[1]))
    classes_counted = torch.unique(torch.argmax(y, dim=-1), return_counts=True)
    for c, samples in zip(classes_counted[0], classes_counted[1]):
        spc[c] = samples

    return spc

def set_threads(num_threads, parallelize):
    if num_threads > 10 and not parallelize:
        # 4 for dask, 4 for dataloader, the rest for torch
        num_workers = 4
        num_threads = num_threads - 4 - 4

    else:
        num_workers = 0
        num_threads = 1

    logger.info(f'num_workers set to {num_workers}')    
    torch.set_num_threads(num_threads)
    logger.info(f'num_threads set to {torch.get_num_threads()}')
    return num_workers

def dataloaders_from_dask(x, y, batch_size, num_workers):
    total_samples = x.shape[0]

    indices = np.arange(total_samples)
    rng = np.random.default_rng(12345)
    rng.shuffle(indices)
    indices_train = indices[:int(np.floor(total_samples * .8))]
    indices_val = indices[int(np.floor(total_samples * .8)):]
    x_train, y_train = x[indices_train], y[indices_train]
    x_val, y_val = x[indices_val], y[indices_val]

    batch_size = min(batch_size, x_train.shape[0])
    x_train = da.from_array(x_train, chunks=(batch_size, x_train.shape[1]))
    x_train = x_train.map_blocks(
        sparse.csr_matrix.toarray, 
        dtype=np.float32)
    y_train = da.from_array(y_train, chunks=(batch_size, y_train.shape[1]))
    train_dataset = DaskBatchDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=None, num_workers=num_workers)

    batch_size = min(batch_size, x_val.shape[0])
    x_val = da.from_array(x, chunks=(batch_size, x_val.shape[1]))
    x_val = x_val.map_blocks(
        sparse.csr_matrix.toarray, 
        dtype=np.float32)
    y_val = da.from_array(y_val, chunks=(batch_size, y_val.shape[1]))
    val_dataset = DaskBatchDataset(x_val, y_val)
    val_dataloader = DataLoader(val_dataset, batch_size=None, num_workers=num_workers)

    return train_dataloader, val_dataloader

def dataloaders_from_dense(x, y, batch_size, num_workers):
    x = torch.from_numpy(x).to(torch.float32)
    y = torch.from_numpy(x).to(torch.float32)
    dataset = TensorDataset(x, y)
    train_dataset, val_dataset = random_split(
        dataset, [0.8, 0.2])
    batch_size = min(batch_size, len(train_dataset))
    leaves_remainder = len(train_dataset) % batch_size == 1
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=leaves_remainder,
        num_workers=num_workers)
    
    batch_size = min(batch_size, len(val_dataset))    
    leaves_remainder = len(val_dataset) % batch_size == 1
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=leaves_remainder,
        num_workers=num_workers)
    
    return train_dataloader, val_dataloader

def fit_torch(
        model: DenseTorch, 
        x: np.array, y: np.array, 
        epochs: int=40, batch_size: int=64, 
        starting_lr: float=0.01, max_lr: float=0.1, momentum: float=0.5, 
        parallelize: bool=True, num_threads: int=1, 
        beta: float=0.8, gamma: float=2.0, class_balance: bool=True, max_cells: int=1_000_000):
    
    num_workers = set_threads(num_threads, parallelize)
    y = to_categorical(y, num_classes=len(model.labels_enc.keys()))    
    total_samples = x.shape[0]
    if total_samples > max_cells:
        train_dataloader, val_dataloader = dataloaders_from_dask(x, y, batch_size, num_workers)

    else:
        train_dataloader, val_dataloader = dataloaders_from_dense(x, y, batch_size, num_workers)

    model.train()
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=starting_lr, 
        momentum=momentum
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=max_lr,
        div_factor=10,
        epochs=epochs,
        steps_per_epoch=len(train_dataloader),
        verbose=False
    )
    loss_function = BalancedLoss(
        loss_type="focal_loss",
        samples_per_class=samples_per_class(y),
        beta=beta, # class-balanced loss beta
        fl_gamma=gamma, # focal loss gamma
        class_balanced=class_balance,
        safe=True
    )
    state_dicts = []
    learning_curve = pd.DataFrame(columns=['loss', 'val_loss', 'lr'])
    for epoch in range(epochs):
        model.train()
        cumulative_loss = 0
        for xb, yb in train_dataloader:
            logits = model(xb)
            logits = torch.clamp(logits, 0, 1)
            loss = loss_function(logits, torch.argmax(yb, dim=-1).to(torch.int64))
            loss.backward()

            cumulative_loss += loss.item()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        cumulative_loss = cumulative_loss / len(train_dataloader)
        model.eval()
        running_vloss = 0.0
        for xb, yb in val_dataloader:
            logits = model(xb)
            logits = torch.clamp(logits, 0, 1)
            val_loss = loss_function(logits, torch.argmax(yb, dim=-1).to(torch.int64)).item()
            running_vloss += val_loss

        val_loss = running_vloss / len(val_dataloader)
        learning_curve.loc[epoch, ['loss', 'val_loss', 'lr']] = cumulative_loss, val_loss, scheduler.get_last_lr()
        state_dicts.append(deepcopy(model.state_dict()))
        
    model.load_state_dict(state_dicts[np.argmin(learning_curve['val_loss'].values)])
    
    return learning_curve

def fit_logreg(model: LogisticRegression, x, y):
    fit = model.fit(
        x, y,
    )
    model.labels_enc = {label: i for i, label in enumerate(model.model.classes_)}
    model.labels_dec = {model.labels_enc[label]: label for label in model.labels_enc.keys()}

    return fit

def fit_trees(model: BoostedTrees, x, y, **fit_kwargs):
    x, x_val, y, y_val = train_test_split(x, y, train_size=0.75, random_state=42)
    if not np.all(np.isin(np.unique(y_val), np.unique(y))):
        # if the validation set contains labels not in the training set, remove them
        x_val = x_val[np.isin(y_val, np.unique(y))]
        y_val = np.array([label for label in y_val if label in np.unique(y)])

    fit = model.model.fit(
        x, y,
        eval_set=[(x_val, y_val)],
        #**fit_kwargs
    )

    return fit

def fit(
        model: Union[DenseTorch, LogisticRegression, DummyClassifier], 
        x: np.array, y: np.array,
        standardize_idx: list=None,
        **fit_kwargs):
    """Args:
        model (Union[DenseTorch, LogisticRegression, DummyClassifier]): Model to be fitted.
        x (np.array): Input data.
        y (np.array): Target data in the shape of a 1-dimensional array of label strings.

    Returns:
        _type_: _description_
    """
    
    # Standardize batches separately if list of idxs per dataset is provided
    if standardize_idx is not None:
        for idx in standardize_idx:
            x[idx] = robust_scale(x[idx], axis=1, with_centering=False, copy=False, unit_variance=True)
    else:
        x = robust_scale(x, axis=1, with_centering=False, copy=False, unit_variance=True)

    y = np.array([model.labels_enc[label] for label in y])
    if isinstance(model, DenseTorch):
        return fit_torch(model, x, y, **fit_kwargs)
    
    elif isinstance(model, LogisticRegression):
        if hasattr(x, 'todense'):
            x = x.todense()
            
        return fit_logreg(model, x, y, **fit_kwargs)
    
    elif isinstance(model, BoostedTrees):
        if hasattr(x, 'todense'):
            x = x.todense()
            
        return fit_trees(model, x, y, **fit_kwargs)

    elif isinstance(model, DummyClassifier):
        if hasattr(x, 'todense'):
            x = x.todense()
            
        return model.fit(x, y)