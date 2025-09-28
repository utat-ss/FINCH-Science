import matplotlib.pyplot as plt
import numpy as np
import os

def plot_fold_from_master(folds_dict, fold_idx, save_dir=None, max_outputs=3):
    """
    Plots losses and predictions for a given fold in the folds_dict.
    
    Parameters:
        folds_dict (dict): The master dictionary with fold data.
        fold_idx (int): Which fold to plot (0-based).
        save_dir (str, optional): Directory to save plots. If None, shows interactively.
        max_outputs (int): Max number of output dims to plot for predictions.
    """
    fold_key = f"fold_{fold_idx}_dict"
    fold_loss_dict = folds_dict[fold_key]['fold_loss_dict']
    fold_predictions_dict = folds_dict[fold_key]['fold_predictions_dict']
    fold_labels_dict = folds_dict[fold_key]['fold_labels_dict']

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # ---------------- 1. LOSS PLOTS ----------------
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    loss_keys = ["total", "reconstruction", "sum", "bounds"]

    for ax, loss_key in zip(axs.flatten(), loss_keys):
        train_loss = fold_loss_dict['train'][loss_key]
        val_loss = fold_loss_dict['val'][loss_key]

        # Make sure they are 1D
        train_loss = np.ravel(train_loss)
        val_loss = np.ravel(val_loss)

        ax.plot(train_loss, label="Train")
        ax.plot(val_loss, label="Validation")
        ax.set_title(f"{loss_key.capitalize()} Loss")
        ax.set_xlabel("Batch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True)

    plt.suptitle(f"Fold {fold_idx} Losses", fontsize=16)
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"fold_{fold_idx}_losses.png"))
        plt.close()
    else:
        plt.show()

    # ---------------- 2. PREDICTIONS vs LABELS ----------------
    train_preds = fold_predictions_dict['train']
    train_labels = fold_labels_dict['train']
    val_preds = fold_predictions_dict['val']
    val_labels = fold_labels_dict['val']

    num_outputs = train_preds.shape[1]
    num_plot_dims = min(num_outputs, max_outputs)
    
    fig, axs = plt.subplots(1, num_plot_dims, figsize=(5 * num_plot_dims, 4))
    if num_plot_dims == 1:
        axs = [axs]

    for dim in range(num_plot_dims):
        # Train points
        axs[dim].scatter(train_labels[:, dim], train_preds[:, dim], alpha=0.4, label='Train', color='blue')
        # Validation points
        axs[dim].scatter(val_labels[:, dim], val_preds[:, dim], alpha=0.4, label='Val', color='orange')
        # y=x line
        min_val = min(train_labels[:, dim].min(), val_labels[:, dim].min())
        max_val = max(train_labels[:, dim].max(), val_labels[:, dim].max())
        axs[dim].plot([min_val, max_val], [min_val, max_val], 'r--')
        
        axs[dim].set_title(f"Pred vs Actual (Output {dim+1})")
        axs[dim].set_xlabel("Actual")
        axs[dim].set_ylabel("Predicted")
        axs[dim].legend()
        axs[dim].grid(True)

    plt.suptitle(f"Fold {fold_idx} Predictions", fontsize=16)
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"fold_{fold_idx}_predictions.png"))
        plt.close()
    else:
        plt.show()