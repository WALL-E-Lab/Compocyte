import sqlite3
import time
import numpy as np
import pandas as pd
import scanpy as sc
import optuna
import os

import torch
import torch.functional as F
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

def train_from_tuned_with_optuna(classifier: HierarchicalClassifier, studies: dict):
    for node in classifier.graph.nodes:
        pass

    # TODO: implement

class Tuner():
    def __init__(self, database_path: str, adata_path: str, hierarchy: dict, root_node: str, obs_names: list):
        self.con = sqlite3.connect(database_path)
        self.cur = self.con.cursor()
        self.adata_path = adata_path
        self.hierarchy = hierarchy
        self.root_node = root_node
        self.obs_names = obs_names
        
    def train_from_tuner(self, save_path: str, adata: sc.AnnData, parallelize=True, max_cells: int=None, stratify_by: str=None, processes: int=None, intermittent_saving=False) -> HierarchicalClassifier:
        classifier = HierarchicalClassifier(
            save_path, 
            root_node=self.root_node, 
            adata=adata, 
            dict_of_cell_relations=self.hierarchy,
            obs_names=self.obs_names,
            intermittent_saving=intermittent_saving)
        
        for node in classifier.graph.nodes:
            n_children = len(list(classifier.graph.successors(node)))
            if n_children >= 1:
                subset = classifier.select_subset(node)
                if len(subset) < 5:
                    continue

                tup = self.get_best_trial(node)
                # No hypopt results exist for this node
                # Use defaults    
                if tup is None:
                    continue
                    
                else:
                    kwargs = {
                        'n_features': tup[0],
                        'hidden_layers': eval(tup[1]),
                        'dropout': tup[2],
                        'epochs': tup[3],
                        'batch_size': tup[4],
                        'starting_lr': tup[5],
                        'max_lr': tup[6],
                        'momentum': tup[7],
                        'beta': tup[8],
                        'gamma': tup[9],
                        'threshold': tup[10],
                    }

                if not hasattr(classifier, 'tuned_kwargs'):
                    classifier.tuned_kwargs = {}

                classifier.tuned_kwargs[node] = kwargs

        if max_cells is not None and stratify_by is not None:
            classifier.introduce_limit(max_cells, stratify_by)
            
        if not parallelize:
            classifier.num_threads = processes

        else:
            classifier.num_threads = 1
            
        classifier.train_all_child_nodes(parallelize=parallelize, processes=processes)
        return classifier
    
    def trial_run(
        self,
        cv_key: str,
        node: str,
        standardize_separately: str=None) -> None:
        
        adata = sc.read_h5ad(self.adata_path)
        for dataset in adata.obs[cv_key].unique():
            train_adata = adata[adata.obs[cv_key] != dataset]
            val_adata = adata[adata.obs[cv_key] == dataset]
            classifier = HierarchicalClassifier(
                'testing', 
                root_node=self.root_node, 
                adata=train_adata, 
                dict_of_cell_relations=self.hierarchy,
                obs_names=self.obs_names)
            
            n_children = len(list(classifier.graph.successors(node)))
            if n_children >= 1:
                subset = classifier.select_subset(node)
                if len(subset) < 5:
                    continue
                    
                features = classifier.run_feature_selection(
                    node=node,
                    overwrite=False,
                    n_features=n_features,
                    max_features=None,
                    min_features=30,
                    test_factor=test_factor)
                classifier.graph.nodes[node]['selected_var_names'] = features
                classifier_type = DenseTorch
                hidden_layers = hidden_layers if isinstance(hidden_layers, list) else eval(hidden_layers)
                if -1 in hidden_layers:
                    classifier_type = BoostedTrees

                classifier.create_local_classifier(
                    node,
                    hidden_layers=hidden_layers,
                    dropout=dropout,
                    batchnorm=True,
                    classifier_type=classifier_type
                )
                features = classifier.graph.nodes[node]['selected_var_names']
                model = classifier.graph.nodes[node]['local_classifier']
                subset = classifier.select_subset(node, features=features)
                x = subset.X
                child_obs = classifier.obs_names[classifier.node_to_depth[node] + 1]
                y = subset.obs[child_obs].values
                if standardize_separately is not None:
                    idx = []
                    for dataset in subset.obs[standardize_separately].unique():
                        idx.append(np.where(subset.obs[standardize_separately] == dataset))

                else:
                    idx = None

                fit(model, x, y, 
                    standardize_idx=idx,
                    epochs=epochs,
                    batch_size=batch_size,
                    starting_lr=starting_lr,
                    max_lr=max_lr,
                    momentum=momentum,
                    beta=beta,
                    gamma=gamma)
                    
            classifier.load_adata(val_adata)
            for node in classifier.graph.nodes:
                if 'local_classifier' not in classifier.graph.nodes[node]:
                    continue
                    
                features = classifier.graph.nodes[node]['selected_var_names']
                model = classifier.graph.nodes[node]['local_classifier']
                subset = classifier.select_subset_prediction(node, features=features, for_trial=True)
                if len(subset) < 5:
                    continue
                    
                x = subset.X
                child_obs = self.obs_names[classifier.node_to_depth[node] + 1]
                y = subset.obs[child_obs].values
                label_enc = model.labels_enc
                y = np.array([label_enc[label] if label in label_enc.keys() else -1 for label in y])
                logits = predict_logits(model, x)
                activations = np.max(logits, axis=1)
                matches = np.argmax(logits, axis=1) == y
                if hasattr(model, 'labels_dec'):
                    child_obs = f'{child_obs}_pred'
                    if child_obs not in classifier.adata.obs.columns:
                        classifier.adata.obs[child_obs] = ''
                        
                    classifier.adata.obs[child_obs] = classifier.adata.obs[child_obs].astype(str)
                    pred = np.argmax(logits, axis=1).astype(int)
                    pred = np.array([model.labels_dec[p] for p in pred])
                    classifier.adata.obs.loc[
                        subset.obs_names,
                        child_obs
                    ] = pred
                    
                for threshold in range(100):
                    threshold /= 100
                    max_correct = len(matches)
                    n_matches = np.sum(matches)
                    correct_positive = matches & (activations > threshold)
                    correct_negative = (~matches) & (activations <= threshold)
                    correct_total = np.sum(correct_positive) + np.sum(correct_negative)
                    performance_per_cv.loc[
                        len(performance_per_cv),
                        ['node', 'threshold', 'n_matches', 'max_correct', 'correct_total']
                    ] = [node, threshold, n_matches, max_correct, correct_total]
                    
        trials = len(adata.obs[cv_key].unique())
        for node in performance_per_cv.node.unique():
            node_performance = performance_per_cv[performance_per_cv.node == node]
            for threshold in node_performance.threshold.unique():
                threshold_performance = node_performance[node_performance.threshold == threshold]
                n_matches = threshold_performance.n_matches.sum()
                correct_total = threshold_performance.correct_total.sum()
                max_total = threshold_performance.max_correct.sum()
                fraction_matches = n_matches / max_total
                fraction_correct = correct_total / max_total
                self.make_entry(
                    node=node,
                    trials=trials,
                    fraction_correct=fraction_correct,
                    fraction_matches=fraction_matches,
                    n_features=n_features,
                    hidden_layers=hidden_layers,
                    dropout=dropout,
                    epochs=epochs,
                    batch_size=batch_size,
                    starting_lr=starting_lr,
                    max_lr=max_lr,
                    momentum=momentum,
                    beta=beta, 
                    gamma=gamma, 
                    threshold=threshold)
        
    def make_db(self) -> None:
        self.cur.execute("""CREATE TABLE IF NOT EXISTS trials(
            node, 
            trials, 
            fraction_correct,
            fraction_matches,
            n_features, 
            hidden_layers, 
            dropout, 
            epochs, 
            batch_size, 
            starting_lr, 
            max_lr, 
            momentum, 
            beta, 
            gamma, 
            threshold,
            t TIMESTAMP)""")
        self.con.commit()
        
    def make_entry(
        self,
        node: str,
        trials: int,
        fraction_correct: float,
        fraction_matches: int,
        n_features: int,
        hidden_layers: str,
        dropout: float,
        epochs: int,
        batch_size: int,
        starting_lr: float,
        max_lr: float,
        momentum: float,
        beta: float, 
        gamma: float, 
        threshold: float) -> None:
        
        for i in range(3):
            try:
                self.cur.execute(f"""
                INSERT INTO trials VALUES
                    ('{node}', {trials}, {fraction_correct}, {fraction_matches}, {n_features}, '{hidden_layers}', {dropout}, {epochs}, {batch_size}, {starting_lr}, {max_lr}, {momentum}, {beta}, {gamma}, {threshold}, DATETIME('now'))
                """)
                self.con.commit()
                
                break
            except sqlite3.OperationalError:
                
                time.sleep(0.01)
        
        
    def get_best_trial(self, node) -> dict:
        res = None
        for i in range(10):
            try:
                res = self.cur.execute(
                    f"""SELECT n_features, hidden_layers, dropout, epochs, batch_size, starting_lr, max_lr, momentum, beta, gamma, threshold 
                    FROM trials 
                    WHERE node == '{node}' 
                    ORDER BY fraction_matches DESC, fraction_correct DESC"""
                )
                
                break

            except sqlite3.OperationalError:                
                time.sleep(0.01)
        
        if res is None:
            tup = None

        else:
            tup = res.fetchone() 
                   
        return tup       
        
    def __del__(self):
        self.con.close()
        