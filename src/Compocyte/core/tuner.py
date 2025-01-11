import sqlite3
import time
import numpy as np
import pandas as pd
import scanpy as sc
from Compocyte.core.hierarchical_classifier import HierarchicalClassifier
from Compocyte.core.models.fit_methods import fit, predict_logits

class Tuner():
    def __init__(self, database_path: str, adata_path: str, hierarchy: dict, root_node: str, obs_names: list):
        self.con = sqlite3.connect(database_path)
        self.cur = self.con.cursor()
        self.adata_path = adata_path
        self.hierarchy = hierarchy
        self.root_node = root_node
        self.obs_names = obs_names
        
    def train_from_tuner(self, save_path: str, adata: sc.AnnData, parallelize=True) -> HierarchicalClassifier:
        classifier = HierarchicalClassifier(
            save_path, 
            root_node=self.root_node, 
            adata=adata, 
            dict_of_cell_relations=self.hierarchy,
            obs_names=self.obs_names)
        
        for node in classifier.graph.nodes:
            n_children = len(list(classifier.graph.successors(node)))
            if n_children >= 1:
                subset = classifier.select_subset(node)
                if len(subset) == 0:
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

                if not hasattr(self, 'tuned_kwargs'):
                    classifier.tuned_kwargs = {}

                classifier.tuned_kwargs[node] = kwargs


        classifier.train_all_child_nodes(parallelize=parallelize)
        return classifier
    
    def trial_run(
        self,
        cv_key: str, 
        n_features: int, 
        hidden_layers: list, 
        dropout: float, 
        epochs: int, 
        batch_size: int, 
        starting_lr: float, 
        max_lr: float, 
        momentum: float, 
        beta: float, 
        gamma: float,
        test_factor: int,
        parallelize: bool=True,
        num_threads=None):
        
        adata = sc.read_h5ad(self.adata_path)
        adata = adata[
            np.random.choice(adata.obs_names, int(len(adata) / test_factor), replace=False)]
        performance_per_cv = pd.DataFrame(columns=['node', 'threshold', 'max_correct', 'correct_total'])
        for dataset in adata.obs[cv_key].unique():
            train_adata = adata[adata.obs[cv_key] == dataset]
            val_adata = adata[adata.obs[cv_key] != dataset]
            classifier = HierarchicalClassifier(
                'testing', 
                root_node=self.root_node, 
                adata=train_adata, 
                dict_of_cell_relations=self.hierarchy,
                obs_names=self.obs_names)
            
            for node in classifier.graph.nodes:
                n_children = len(list(classifier.graph.successors(node)))
                if n_children >= 1:
                    subset = classifier.select_subset(node)
                    if len(subset) == 0:
                        continue
                        
                    features = classifier.run_feature_selection(
                        node=node,
                        overwrite=False,
                        n_features=n_features,
                        max_features=None,
                        min_features=30,
                        test_factor=test_factor)
                    classifier.graph.nodes[node]['selected_var_names'] = features
                    classifier.create_local_classifier(
                        node,
                        hidden_layers=hidden_layers if isinstance(hidden_layers, list) else eval(hidden_layers),
                        dropout=dropout,
                        batchnorm=True
                    )
                    features = classifier.graph.nodes[node]['selected_var_names']
                    model = classifier.graph.nodes[node]['local_classifier']
                    subset = classifier.select_subset(node, features=features)
                    x = subset.X
                    child_obs = classifier.obs_names[classifier.node_to_depth[node] + 1]
                    y = subset.obs[child_obs].values
                    fit(model, x, y, 
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
                subset = classifier.select_subset_prediction(node, features=features)
                if len(subset) == 0:
                    continue
                    
                x = subset.X
                child_obs = self.obs_names[classifier.node_to_depth[node] + 1]
                y = subset.obs[child_obs].values
                label_enc = model.labels_enc
                y = np.array([label_enc[label] if label in label_enc.keys() else -1 for label in y])
                logits = predict_logits(model, x)
                activations = np.max(logits, axis=1)
                matches = np.argmax(logits, axis=1) == y
                for threshold in range(100):
                    threshold /= 100
                    max_correct = len(matches)
                    correct_positive = matches & (activations > threshold)
                    correct_negative = (~matches) & (activations <= threshold)
                    correct_total = np.sum(correct_positive) + np.sum(correct_negative)   
                    performance_per_cv.loc[
                        len(performance_per_cv),
                        ['node', 'threshold', 'max_correct', 'correct_total']
                    ] = [node, threshold, max_correct, correct_total]
                    
        trials = len(adata.obs[cv_key].unique())
        for node in performance_per_cv.node.unique():
            node_performance = performance_per_cv[performance_per_cv.node == node]
            for threshold in node_performance.threshold.unique():
                threshold_performance = node_performance[node_performance.threshold == threshold]
                correct_total = threshold_performance.correct_total.sum()
                max_total = threshold_performance.max_correct.sum()
                fraction_correct = correct_total / max_total
                self.make_entry(
                    node=node,
                    trials=trials,
                    fraction_correct=fraction_correct,
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
                    ('{node}', {trials}, {fraction_correct}, {n_features}, '{hidden_layers}', {dropout}, {epochs}, {batch_size}, {starting_lr}, {max_lr}, {momentum}, {beta}, {gamma}, {threshold}, DATETIME('now'))
                """)
                self.con.commit()
                
                break
            except sqlite3.OperationalError:
                
                time.sleep(0.01)
        
        
    def get_best_trial(self, node) -> dict:
        for i in range(3):
            try:
                res = self.cur.execute(
                    f"""SELECT n_features, hidden_layers, dropout, epochs, batch_size, starting_lr, max_lr, momentum, beta, gamma, threshold 
                    FROM trials 
                    WHERE node == '{node}' 
                    ORDER BY fraction_correct DESC"""
                )
                
                break
            except sqlite3.OperationalError:
                
                time.sleep(0.01)
        
        tup = res.fetchone()        
        return tup       
        
    def __del__(self):
        self.con.close()
        