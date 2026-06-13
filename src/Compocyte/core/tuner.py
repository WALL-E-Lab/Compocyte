import sqlite3
import time
import numpy as np
import pandas as pd
import scanpy as sc
import optuna
import os

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from Compocyte.core.hierarchical_classifier import HierarchicalClassifier
from Compocyte.core.models.dense_torch import DenseTorch
from Compocyte.core.models.trees import BoostedTrees
from Compocyte.core.models.fit_methods import dataloaders_from_dense, fit, predict_logits, samples_per_class, to_categorical

def tune_with_optuna(
        storage_path: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        cores: int,
        n_trials=50, n_startup_trials=5
        ) -> optuna.Study:
    
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    y = to_categorical(y_train, num_classes=len(np.unique(y_train)))
    x = torch.from_numpy(X_train).to(torch.float32)
    y = torch.from_numpy(y).to(torch.float32)
    y_val = to_categorical(y_val, num_classes=len(np.unique(y_train)))
    x_val = torch.from_numpy(X_val).to(torch.float32)
    y_val = torch.from_numpy(y_val).to(torch.float32)
    def objective(trial):
        model = DenseTorch(
            labels=list(np.unique(y_train)), 
            n_input=X_train.shape[1],
            n_output=len(np.unique(y_train)),
            hidden_layers=eval(trial.suggest_categorical('hidden_layers', ['[]', '[64, 64]', '[128, 128, 128, 128]', '[256, 256, 256, 256, 256, 256, 256, 256]'])),
            dropout=trial.suggest_float('dropout', 0, 0.5),
        )        
        epochs=trial.suggest_int('epochs', 10, 100)
        batch_size=trial.suggest_categorical('batch_size', [64, 256, 512])
        starting_lr=trial.suggest_float('starting_lr', 1e-5, 1e-3, log=True)
        max_lr=trial.suggest_float('max_lr', 1e-3, 1e-1, log=True)
        momentum=trial.suggest_float('momentum', 0.4, 0.9)
        beta=trial.suggest_float('beta', 0.8, 0.999)
        gamma=trial.suggest_float('gamma', 1, 4)
        
        train_dataset = TensorDataset(x, y)
        val_dataset = TensorDataset(x_val, y_val)
        batch_size = min(batch_size, len(train_dataset))
        leaves_remainder = len(train_dataset) % batch_size == 1
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=leaves_remainder,
            num_workers=1)
        
        batch_size = min(batch_size, len(val_dataset))    
        leaves_remainder = len(val_dataset) % batch_size == 1
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=leaves_remainder,
            num_workers=1)
        num_batches = len(train_dataloader)
        num_batches_val = len(val_dataloader)

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
            steps_per_epoch=num_batches,
            pct_start=0.3
        )
        # Loss adapted from https://github.com/fcakyon/balanced-loss
        effective_num = 1.0 - np.power(beta, samples_per_class(y))
        # Avoid division by 0 error for test cases without all labels present.
        effective_num_classes = np.sum(effective_num != 0)
        effective_num[effective_num == 0] = np.inf
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * effective_num_classes
        weights = torch.tensor(weights).float()

        for epoch in range(epochs):                
            model.train()
            cumulative_loss = 0
            for xb, yb in train_dataloader:
                logits = model(xb)            
                # Used a stable cross-entropy based focal loss implementation instead 
                # to avoid numerical instability with large gamma values.
                # Using the existing cross-entropy implementation to avoid exponetiating
                # large logits directly.
                ce = F.cross_entropy(
                    logits,
                    torch.argmax(yb, dim=-1).to(torch.int64),
                    reduction='none',
                    weight=weights
                )
                pt = torch.exp(-ce)
                loss = ((1 - pt) ** gamma) * ce
                loss = loss.mean()
                loss.backward()

                cumulative_loss += loss.item()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            cumulative_loss = cumulative_loss / num_batches
            model.eval()
            running_vloss = 0.0
            for xb, yb in val_dataloader:
                logits = model(xb)
                ce = F.cross_entropy(
                    logits,
                    torch.argmax(yb, dim=-1).to(torch.int64),
                    reduction='none',
                    weight=weights
                )
                pt = torch.exp(-ce)
                val_loss = ((1 - pt) ** gamma) * ce
                val_loss = val_loss.mean().item()
                running_vloss += val_loss

            val_loss = running_vloss / num_batches_val
            if epoch > epochs * 0.3: # val loss is only reported after 30% of epochs have passed to allow for warmup
                trial.report(val_loss, epoch)

        return val_loss

    storage = f"sqlite:///{storage_path}" if storage_path is not None else None
    study = optuna.create_study(
        storage=storage,
        load_if_exists=False,
        direction="minimize",
        pruner=optuna.pruners.PercentilePruner(
            percentile=25.0,
            n_startup_trials=n_startup_trials,
            n_warmup_steps=0,
            interval_steps=1
        ),
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=cores-1)

    return study

def train_from_tuned_with_optuna(
        db_path, 
        save_path,
        root_node, 
        adata, 
        hierarchy, 
        obs_names, 
        processes,
        intermittent_saving=False, 
        parallelize=False,
        max_cells=None,
        stratify_by=None) -> HierarchicalClassifier:
    
    classifier = HierarchicalClassifier(
        save_path, 
        root_node=root_node, 
        adata=adata,
        dict_of_cell_relations=hierarchy,
        obs_names=obs_names,
        intermittent_saving=intermittent_saving
    )
    if max_cells is not None and stratify_by is not None:
            classifier.introduce_limit(max_cells, stratify_by)

    for node in classifier.graph.nodes:
        n_children = len(list(classifier.graph.successors(node)))
        if n_children >= 1:
            subset = classifier.select_subset(node)
            if len(subset) < 5:
                continue

            if os.path.exists(db_path.format(node)):
                study = optuna.load_study(storage=f"sqlite:///{db_path.format(node)}", study_name=None)
                kwargs = study.best_trial.params
                kwargs['hidden_layers'] = eval(kwargs['hidden_layers'])

            else:
                # If no tuning results exist for this node, skip it and use defaults
                continue

            if not hasattr(classifier, 'tuned_kwargs'):
                classifier.tuned_kwargs = {}

            classifier.tuned_kwargs[node] = kwargs
        
    if not parallelize:
        classifier.num_threads = processes

    else:
        classifier.num_threads = 1
        
    classifier.train_all_child_nodes(parallelize=parallelize, processes=processes)
    return classifier