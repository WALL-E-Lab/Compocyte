from copy import deepcopy
import os
from typing import Union
import numpy as np
import pandas as pd
import torch
import logging
import torch.utils
import torch.utils.data
from Compocyte.core.models.dense_torch import DenseTorch
from Compocyte.core.models.dummy_classifier import DummyClassifier
from Compocyte.core.models.log_reg import LogisticRegression
from Compocyte.core.tools import z_transform_properties
from keras.utils import to_categorical
from balanced_loss import Loss as BalancedLoss

logger = logging.getLogger(__name__)

def predict_logits(model, x):
    if hasattr(x, 'todense'):
        x = x.todense()

    
    x = z_transform_properties(x)
    if isinstance(model, DenseTorch):
        logits = model.predict_logits(x)
    
    elif isinstance(model, LogisticRegression):
        logits = model.predict_logits(x)

    elif isinstance(model, DummyClassifier):
        logits = model.predict_logits(x)

    else:
        raise Exception('Unknown classifier type.')

    return logits

def predict(model, x, threshold=None):
    if hasattr(x, 'todense'):
        x = x.todense()

    x = z_transform_properties(x)
    if isinstance(model, DenseTorch):
        logits = model.predict_logits(x)
    
    elif isinstance(model, LogisticRegression):
        logits = model.predict_logits(x)

    elif isinstance(model, DummyClassifier):
        return model.predict(x)
    
    else:
        raise Exception('Unknown classifier type')
    
    max_activation = np.max(logits, axis=1)
    pred = np.argmax(logits, axis=1).astype(int)
    pred = np.array([model.label_dec[p] for p in pred])
    if threshold is not None:
        pred[max_activation <= threshold] = ''

    return pred

def samples_per_class(y):
    spc = list(torch.zeros(y.shape[1]))
    classes_counted = torch.unique(torch.argmax(y, dim=-1), return_counts=True)
    for c, samples in zip(classes_counted[0], classes_counted[1]):
        spc[c] = samples

    return spc

def fit_torch(
        model: DenseTorch, 
        x: np.array, y: np.array, 
        epochs: int, batch_size: int, 
        starting_lr: float, max_lr: float, momentum: float, 
        parallelized: bool, 
        beta: float=0.8, gamma: float=2.0, class_balance: bool=True):
    
    torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
    logger.info(f'num_threads set to {torch.get_num_threads()}')
    logger.info(f'OMP_NUM_THREADS set to {os.environ["OMP_NUM_THREADS"]}')

    y = to_categorical(y, num_classes=len(model.labels_enc.keys()))
    x = torch.Tensor(x)
    y = torch.Tensor(y)
    dataset = torch.utils.data.TensorDataset(x, y)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [0.8, 0.2])
    leaves_remainder = len(train_dataset) % batch_size == 1
    num_workers = min(os.cpu_count(), 2)
    if parallelized:
            num_workers = 0

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=leaves_remainder,
        num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=len(val_dataset))

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
        for xb, yb in val_dataloader:
            logits = model(xb)
            logits = torch.clamp(logits, 0, 1)
            val_loss = loss_function(logits, torch.argmax(yb, dim=-1).to(torch.int64)).item()

        learning_curve.loc[epoch, ['loss', 'val_loss', 'lr']] = cumulative_loss, val_loss, scheduler.get_last_lr()
        state_dicts.append(deepcopy(model.state_dict()))
        
    model.load_state_dict(state_dicts[np.argmin(learning_curve['val_loss'].values)])
    
    return learning_curve

def fit_logreg(model: LogisticRegression, x, y):
    return model.model.fit(x, y)

def fit(
        model: Union[DenseTorch, LogisticRegression, DummyClassifier], 
        x: np.array, y: np.array, 
        **fit_kwargs):
    """Args:
        model (Union[DenseTorch, LogisticRegression, DummyClassifier]): Model to be fitted.
        x (np.array): Input data.
        y (np.array): Target data in the shape of a 1-dimensional array of label strings.

    Returns:
        _type_: _description_
    """

    if hasattr(x, 'todense'):
        x = x.todense()
        
    x = z_transform_properties(x)
    y = np.array([model.label_enc[label] for label in y])
    if isinstance(model, DenseTorch):
        return fit_torch(model, x, y, **fit_kwargs)
    
    elif isinstance(model, LogisticRegression):
        return fit_logreg(model, x, y, **fit_kwargs)

    elif isinstance(model, DummyClassifier):
        return model.fit(x, y)