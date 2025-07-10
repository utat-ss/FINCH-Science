import pandas as pd
import numpy as np
from NeuralNetwork_Defs import train_Network

df = pd.read_csv('sampled_csrs2025.csv')
labels_df = df.iloc[:, 2:5]

features_df = df.iloc[:, 8:]

#optional for debugging
features_df = features_df.apply(pd.to_numeric, errors='coerce')

#features_df = features_df.dropna()
#labels_df   = labels_df.loc[features_df.index]

X = features_df.values.astype(np.float32)
y = labels_df.values.astype(np.float32)
data_array = np.hstack([X, y])
num_features = X.shape[1]
num_labels   = y.shape[1]

cfg_dataset = {
    "idx_ab_tuple":    (num_features, num_features + num_labels),
    "idx_range_tuple": (0, num_features)
}

cfg_NN = {
    "model_type": "MLP",
    "input_dim":  num_features,
    "output_dim": num_labels,
    "hidden_layers": [64, 32],
    "linear_activation_list": ["ReLU", "ReLU", "Identity"],
}
#========================= edit learning data here
cfg_train = {
    "batch_size": 32,
    "seperation_ratios": (0.7, 0.85),
    "lr": 1e-4,
    "num_epochs": 20
}
cfg_plots = {
    "plot_losses": True,
    "plot_errors": True,
    "plot_separately": True
}
if __name__ == "__main__":
    model, history, test_loss = train_Network(
        cfg_NN,
        cfg_dataset,
        cfg_train,
        cfg_plots,
        data_array
    )
    print("Test loss:", test_loss)