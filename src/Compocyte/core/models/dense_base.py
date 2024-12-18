import os
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset
from balanced_loss import Loss
import logging
logger = logging.getLogger(__name__)

class DenseBase():
    """
    """

    def standardize(self, x):
        # standardize to mean 0 and std 1 within gene/feature
        std = x.std(dim=0)
        mean = x.mean(dim=0)
        x -= mean
        x /= std
        x[x.isnan()] = 0
        """x -= torch.min(x)
        x /= torch.max(x)"""

        return x

    def fit(
        self, 
        x, 
        y,
        batch_size=40,
        epochs=40,
        validation_data=None,
        plot_live=False,
        parallelized=False,
        class_balance=False,
        beta=0.999,
        gamma=2,
        num_threads=None,
        max_lr=0.1):
        """x is torch.Tensor with shape (n_cells, n_features)
        y is torch.Tensor with shape (n_cells, n_output), containing onehot encoding"""

        torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
        logger.info(f'num_threads set to {torch.get_num_threads()}')
        logger.info(f'OMP_NUM_THREADS set to {os.environ["OMP_NUM_THREADS"]}')

        if plot_live:
            from IPython.display import clear_output

        if hasattr(self, 'imported') and self.imported:
            return getattr(self.module, self.fit_function)(x=x, y=y, batch_size=batch_size, epochs=epochs)

        self.train()
        x, y = torch.Tensor(x).to(self.device), torch.Tensor(y).to(self.device)
        #x = self.standardize(x)
        #if torch.max(y) > 1: # test if output is counts (autencoder) or categories
        #    y = self.standardize(y)

        n_cells = x.shape[0]
        if validation_data is not None:
            x_val, y_val = validation_data
            x_val, y_val = torch.Tensor(x_val).to(self.device), torch.Tensor(y_val).to(self.device)
            #x_val = self.standardize(x_val)
            #if torch.max(y_val) > 1: # test if output is counts or categories
            #    y_val = self.standardize(y_val)

            x_val.shape[0]

        if self.l2_reg_input:
            weight_decay = 1e-5

        else:
            weight_decay = 0

        optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=self.learning_rate, 
            momentum=self.momentum,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=self.learning_rate * 10,
            div_factor=10,
            epochs=epochs,
            steps_per_epoch=((n_cells - 1) // batch_size + 1),
            verbose=False
        )
        history = {
            'val_accuracy': [],
            'val_loss': [],
            'accuracy': [],
            'loss': [],
            'lr': [],
            'state_dicts': []}
        counter_stopping = 0
        dataset = TensorDataset(x, y)
        leaves_remainder = len(dataset) % batch_size == 1
        num_workers = min(os.cpu_count(), 2)
        if parallelized:
            num_workers = 0

        logger.info(f'num_workers for DataLoader set to {num_workers}')
        train_data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=leaves_remainder,
            num_workers=num_workers
        )
        epoch_training_times = []
        epoch_total_times = []
        batch_training_times = []
        samples_per_class = list(torch.zeros(y_val.shape[1]))
        classes_counted = torch.unique(torch.argmax(y, dim=-1), return_counts=True)
        for c, samples in zip(classes_counted[0], classes_counted[1]):
            samples_per_class[c] = samples
            
        focal_loss = Loss(
            loss_type="focal_loss",
            samples_per_class=samples_per_class,
            beta=beta, # class-balanced loss beta
            fl_gamma=gamma, # focal loss gamma
            class_balanced=class_balance,
            safe=True
        )
        logger.info(f'Number of batches: {len(train_data_loader)}')
        logger.info(f'Samples per class: {samples_per_class}')
        logger.info(f'Class balanced loss: {class_balance}')

        for epoch in range(epochs):
            t0_epoch_total = time.time()
            self.eval()
            history['state_dicts'].append(deepcopy(self.state_dict()))
            # Record loss and accuracy
            if validation_data is not None:
                to_minimize = 'val_loss'                 
                pred_val = self(x_val)
                pred_val = torch.clamp(pred_val, 0, 1)
                pred_int = np.argmax(pred_val.detach().cpu().numpy(), axis=-1)
                y_int = np.argmax(y_val.detach().cpu().numpy(), axis=-1)
                val_accuracy = np.mean(pred_int == y_int) * 100
                try:
                    if class_balance:
                        val_loss = focal_loss(pred_val, torch.Tensor(y_int).to(torch.int64)).item()

                    else:
                        val_loss = self.loss_function(pred_val, y_val).item()

                except RuntimeError as re:
                    print(pred_val)
                    print(y_val)
                    print(type(pred_val))
                    print(type(y_val))
                    print(pred_val.shape)
                    print(y_val.shape)
                    print(re)
                    raise

                del pred_val
                history['val_accuracy'].append(val_accuracy)
                history['val_loss'].append(val_loss)

            else:
                to_minimize = 'loss'

            pred = self(x)
            pred = torch.clamp(pred, 0, 1)
            pred_int = np.argmax(pred.detach().cpu().numpy(), axis=-1)
            y_int = np.argmax(y.detach().cpu().numpy(), axis=-1)
            accuracy = np.mean(pred_int == y_int) * 100
            loss = self.loss_function(pred, y).item()
            del pred
            history['accuracy'].append(accuracy)
            history['loss'].append(loss)
            history['lr'].append(scheduler.get_last_lr()[0])

            self.train()
            t0_epoch_training = time.time()
            for xb, yb in train_data_loader:
                t0_batch = time.time()
                pred = self(xb)
                pred = torch.clamp(pred, 0, 1)
                if class_balance:
                    loss = focal_loss(pred, torch.argmax(yb, dim=-1).to(torch.int64))

                else:
                    loss = self.loss_function(pred, yb)

                try:
                    loss.backward()

                except RuntimeError as re:
                    print(re)
                    print(pred)
                    print(yb)
                    print(history['loss'])
                    print(torch.any(torch.isnan(pred)))
                    print(torch.any(torch.isnan(torch.argmax(yb, dim=-1).to(torch.int64))))
                    print(loss.item())
                    print(re)
                    raise

                del xb, yb, pred, loss
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                batch_training_times.append(time.time() - t0_batch)

            epoch_training_times.append(time.time() - t0_epoch_training)
            epoch_total_times.append(time.time() - t0_epoch_total)
            if plot_live:
                clear_output()
                fig, ax_left = plt.subplots()
                ax_left.plot(history['lr'], label='lr', color='green')
                ax_left.set_ylabel('model lr')
                ax_right = ax_left.twinx()
                ax_right.plot(history[to_minimize], label=to_minimize, color='orange')
                ax_right.set_ylabel(to_minimize)
                
                plt.legend(['model lr', to_minimize], loc='upper left')
                plt.show()

            best_index = np.argmin(history[to_minimize])
            patience = 5
            is_plateau = (history[to_minimize][-1] - history[to_minimize][best_index]) <= 0 
            if self.early_stopping and is_plateau:
                counter_stopping += 1
                if counter_stopping == patience:
                    break

            else:
                counter_stopping = 0
                
        logger.info(f'Mean time per batch: {np.mean(batch_training_times)} seconds')
        logger.info(f'Mean time per epoch of training: {np.mean(epoch_training_times)} seconds')
        logger.info(f'Mean time per epoch in total: {np.mean(epoch_total_times)} seconds')

        self.load_state_dict(history['state_dicts'][np.argmin(history[to_minimize])])
        del history['state_dicts']
        return history

    def plot_training(self, history, to_plot):
        plt.plot(history[f'{to_plot}'])
        plt.plot(history[f'val_{to_plot}'])
        plt.title(f'model {to_plot}')
        plt.ylabel(f'{to_plot}')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()