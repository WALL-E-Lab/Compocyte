from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import StandardScaler
import numpy as np

    # def _LRClassifier(indata, labels, C, solver, max_iter, n_jobs, **kwargs) -> LogisticRegression:
    #     """
    #     For internal use. Get the logistic Classifier.
    #     """
    #     no_cells = len(labels)
    #     if solver is None:
    #         solver = 'sag' if no_cells>50000 else 'lbfgs'
    #     elif solver not in ('liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'):
    #         raise ValueError(f"🛑 Invalid `solver`, should be one of `'liblinear'`, `'lbfgs'`, `'newton-cg'`, `'sag'`, and `'saga'`")
    #     logger.info(f"🏋️ Training data using logistic regression")
    #     if (no_cells > 100000) and (indata.shape[1] > 10000):
    #         logger.warn(f"⚠️ Warning: it may take a long time to train this dataset with {no_cells} cells and {indata.shape[1]} genes, try to downsample cells and/or restrict genes to a subset (e.g., hvgs)")
    #     classifier = LogisticRegression(C = C, solver = solver, max_iter = max_iter, multi_class = 'ovr', n_jobs = n_jobs, **kwargs)
    #     classifier.fit(indata, labels)
    #     return classifier


    # def _SGDClassifier(indata, labels,
    #                alpha, max_iter, n_jobs,
    #                mini_batch, batch_number, batch_size, epochs, balance_cell_type, **kwargs) -> SGDClassifier:
    #     """
    #     For internal use. Get the SGDClassifier.
    #     """
    #     classifier = SGDClassifier(loss = 'log', alpha = alpha, max_iter = max_iter, n_jobs = n_jobs, **kwargs)
    #     if not mini_batch:
    #         logger.info(f"🏋️ Training data using SGD logistic regression")
    #         if (len(labels) > 100000) and (indata.shape[1] > 10000):
    #             logger.warn(f"⚠️ Warning: it may take a long time to train this dataset with {len(labels)} cells and {indata.shape[1]} genes, try to downsample cells and/or restrict genes to a subset (e.g., hvgs)")
    #         classifier.fit(indata, labels)
    #     else:
    #         logger.info(f"🏋️ Training data using mini-batch SGD logistic regression")
    #         no_cells = len(labels)
    #         if no_cells < 10000:
    #             logger.warn(f"⚠️ Warning: the number of cells ({no_cells}) is not big enough to conduct a proper mini-batch training. You may consider using traditional SGD classifier (mini_batch = False)")
    #         if no_cells <= batch_size:
    #             raise ValueError(f"🛑 Number of cells ({no_cells}) is fewer than the batch size ({batch_size}). Decrease `batch_size`, or use SGD directly (mini_batch = False)")
    #         no_cells_sample = min([batch_number*batch_size, no_cells])
    #         starts = np.arange(0, no_cells_sample, batch_size)
    #         if balance_cell_type:
    #             celltype_freq = np.unique(labels, return_counts = True)
    #             len_celltype = len(celltype_freq[0])
    #             mapping = pd.Series(1 / (celltype_freq[1]*len_celltype), index = celltype_freq[0])
    #             p = mapping[labels].values
    #         for epoch in range(1, (epochs+1)):
    #             logger.info(f"⏳ Epochs: [{epoch}/{epochs}]")
    #             if not balance_cell_type:
    #                 sampled_cell_index = np.random.choice(no_cells, no_cells_sample, replace = False)
    #             else:
    #                 sampled_cell_index = np.random.choice(no_cells, no_cells_sample, replace = False, p = p)
    #             for start in starts:
    #                 classifier.partial_fit(indata[sampled_cell_index[start:start+batch_size]], labels[sampled_cell_index[start:start+batch_size]], classes = np.unique(labels))
    #     return classifier

class LogRegWrapper():

    def __init__():
        pass 


    def _to_array(_array_like) -> np.ndarray:
        """
        For internal use. Turn an array-like object into an array.
        """
        if isinstance(_array_like, pd.DataFrame):
            return _array_like.values
        elif isinstance(_array_like, spmatrix):
            return _array_like.toarray()
        elif isinstance(_array_like, np.matrix):
            return np.array(_array_like)
        elif isinstance(_array_like, np.ndarray):
            return _array_like
        else:
            raise ValueError(f"🛑 Please provide a valid array-like object as input")

    def _data_preparation(X, labels):
        '''mostly taken from corresponding celltypist method, expects AnnData object as input'''
        
        if isinstance(X, AnnData) or (isinstance(X, str) and X.endswith('.h5ad')):
            adata = sc.read(X) if isinstance(X, str) else X
            adata.var_names_make_unique()
            if adata.X.min() < 0:
                logger.info("👀 Detected scaled expression in the input data, will try the .raw attribute")
                try:
                    indata = adata.raw.X
                    genes = adata.raw.var_names
                except Exception as e:
                    raise Exception(f"🛑 Fail to use the .raw attribute in the input object. {e}")
            else:
                indata = adata.X
                genes = adata.var_names
            if isinstance(labels, str) and (labels in adata.obs):
                labels = adata.obs[labels]
            else:
                labels = _to_vector(labels)

        return indata, labels, genes

    def train(self, x, y, feature_selection=True, top_genes=1500, **kwargs):
        x = x.copy()
        #feature selection
        indata = _to_array(indata)
        labels = np.array(labels)
        genes = np.array(genes)

        scaler = StandardScaler()
        indata = scaler.fit_transform(indata[:, ~flag] if flag.sum() > 0 else indata)
        indata[indata > 10] = 10

        if feature_selection:
            logger.info(f"🔎 Selecting features")
            if len(genes) <= top_genes:
                raise ValueError(f"🛑 The number of genes ({len(genes)}) is fewer than the `top_genes` ({top_genes}). Unable to perform feature selection")
            gene_index = np.argpartition(np.abs(classifier.coef_), -top_genes, axis = 1)[:, -top_genes:]
            gene_index = np.unique(gene_index)
            logger.info(f"🧬 {len(gene_index)} features are selected")
            genes = genes[gene_index]
            #indata = indata[:, gene_index]
            logger.info(f"🏋️ Starting the second round of training")
            if use_SGD:
                classifier = _SGDClassifier(indata = indata[:, gene_index], labels = labels, alpha = alpha, max_iter = max_iter, n_jobs = n_jobs, mini_batch = mini_batch, batch_number = batch_number, batch_size = batch_size, epochs = epochs, balance_cell_type = balance_cell_type, **kwargs)
            else:
                classifier = _LRClassifier(indata = indata[:, gene_index], labels = labels, C = C, solver = solver, max_iter = max_iter, n_jobs = n_jobs, **kwargs)
            scaler.mean_ = scaler.mean_[gene_index]
            scaler.var_ = scaler.var_[gene_index]
            scaler.scale_ = scaler.scale_[gene_index]
            scaler.n_features_in_ = len(gene_index)
        #model finalization
        classifier.features = genes

        self.model = LR.fit(x, y) 
    
    def validate():
        pass 

    def predict(self, x):
        x = x.copy()
        pred = self.model.predict_proba()

        print(f"calling from predict in LogRegWrapper: \n pred vec is \n {pred}")

        return pred 
