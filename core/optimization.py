import pandas as pd
import random
from .neural_network import NeuralNetwork
from .node_memory import NodeMemory

def grid_search(
    adata, 
    obs_name,
    options_scVI = [10, 20, 50],
    options_n_neurons = [64, 128, 256],
    options_n_layers = [2, 5, 10],
    options_dropout = [0.1, 0.2, 0.4, 0.6],
    options_learning_rate = [0.002, 0.001, 0.005],
    options_momentum = [0.5, 0.9],
    options_batch_size = [20, 40, 60],
    options_batch_norm = [True, False],
    options_l2_reg = [True, False],
    options_leakiness_ReLU = [0.1, 0.2],
    random_tries = 10,
):
    adata = adata.clone()
    performance_df = pd.DataFrame(columns=[
        'train_accuracy',
        'test_accuracy',
        'scVI_dimensions',
        'layer_structure',
        'dropout',
        'learning_rate',
        'momentum',
        'batch_size',
        'batch_norm',
        'l2_reg',
        'leakiness_ReLU'
    ])
    for _ in range(random_tries):
        scVI_dim = random.sample(options_scVI, 1)[0]
        dummy_node_memory = NodeMemory(adata.obsm[f'X_scVI{scVI_dim}'], adata.obs[obs_name])
        dummy_node_memory._set_processed_input_data()
        output_len = len(adata.obs[obs_name].unique())
        train_index = random.sample(range(len(adata)), int(len(adata) * 0.9))
        test_index = [i for i in range(len(adata)) if not i in train_index]
        X_train, y_train_int, y_train_onehot = dummy_node_memory._get_indexed_data(train_index)
        X_test, y_test_int, y_test_onehot = dummy_node_memory._get_indexed_data(test_index)
        layer_structure = []
        for i in range(random.sample(options_n_layers, 1)[0]):
            layer_structure.append(
                random.sample(options_n_neurons, 1)[0]
            )

        learning_rate = random.sample(options_learning_rate, 1)[0]
        momentum = random.sample(options_momentum, 1)[0]
        dropout = random.sample(options_dropout, 1)[0]
        batch_size = random.sample(options_batch_size, 1)[0]
        batch_norm = random.sample(options_batch_norm, 1)[0]
        l2_reg = random.sample(options_l2_reg, 1)[0]
        leakiness_ReLU = random.sample(options_leakiness_ReLU, 1)[0]
        current_NN = NeuralNetwork(
            X_train,
            y_train_onehot,
            output_len, 
            list_of_hidden_layer_nodes=layer_structure,
            activation_function='relu',
            learning_rate=learning_rate,
            momentum=momentum,
            dropout=dropout,
            batch_size=batch_size,
            batch_norm=batch_norm,
            l2_reg=l2_reg,
            leakiness_ReLU=leakiness_ReLU,
            loss_function='categorical_crossentropy',
            epochs=500)

        current_NN.train()
        train_accuracy = current_NN.validate(X_train, y_train_int)
        test_accuracy = current_NN.validate(X_test, y_test_int) 
        parameters = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'scVI_dimensions': scVI_dim,
            'layer_structure': str(layer_structure),
            'dropout': dropout,
            'learning_rate': learning_rate,
            'momentum': momentum,
            'batch_size': batch_size,
            'batch_norm': batch_norm,
            'l2_reg': l2_reg, 
            'leakiness_ReLU': leakiness_ReLU
        }
        for key, value in parameters.items():
            index = len(performance_df)
            performance_df.loc[index, key] = value
    
    return performance_df