import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

# Define the path to your history file
HISTORY_PATH = "./playlist_model_ckpts/history.pkl"
GRAPHS_DIR = "graphs"  # Name of the directory to save graphs

def plot_history(file_path):
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        print("Make sure you have run the training script at least once.")
        return

    # Load the pickle file
    print(f"Loading history from {file_path}...")
    with open(file_path, "rb") as f:
        history = pickle.load(f)

    # Extract metrics using the NEW keys
    t_loss = history.get("train_loss", [])
    v_loss = history.get("val_loss", [])
    
    t_r_all = history.get("train_r_prec", [])
    v_r_all = history.get("val_r_prec", [])
    
    t_r_hidden = history.get("train_r_prec_hidden", [])
    v_r_hidden = history.get("val_r_prec_hidden", [])

    if not t_loss:
        print("⚠️ History file is empty or missing 'train_loss'.")
        return

    epochs = range(1, len(t_loss) + 1)

    # Create a figure with THREE side-by-side subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))

    # Plot 1: Loss 
    ax1.plot(epochs, t_loss, label='Training Loss', color='blue', marker='.')
    ax1.plot(epochs, v_loss, label='Validation Loss', color='red', marker='.')

    # Find and annotate the best validation loss (min)
    if v_loss:
        best_loss_idx = np.argmin(v_loss)
        best_loss_val = v_loss[best_loss_idx]
        best_epoch = epochs[best_loss_idx]

        ax1.scatter(best_epoch, best_loss_val, color='black', s=100, zorder=5)
        text_offset = (max(v_loss) - min(v_loss)) * 0.05 if len(v_loss) > 1 else 0.1
        ax1.annotate(f'Best: {best_loss_val:.4f}',
                     xy=(best_epoch, best_loss_val),
                     xytext=(best_epoch, best_loss_val + text_offset),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     horizontalalignment='center')

    ax1.set_title('Model Loss (Lower is Better)')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plot 2: R-Precision (ALL)
    if t_r_all and v_r_all:
        ax2.plot(epochs, t_r_all, label='Train R-Prec (All)', color='green', marker='.')
        ax2.plot(epochs, v_r_all, label='Val R-Prec (All)', color='orange', marker='.')

        # Find best (max)
        if v_r_all:
            best_idx = np.argmax(v_r_all)
            best_val = v_r_all[best_idx]
            best_epoch = epochs[best_idx]

            ax2.scatter(best_epoch, best_val, color='black', s=100, zorder=5)
            text_offset = (max(v_r_all) - min(v_r_all)) * 0.05 if len(v_r_all) > 1 else 0.01
            ax2.annotate(f'Best: {best_val:.4f}',
                         xy=(best_epoch, best_val),
                         xytext=(best_epoch, best_val - text_offset), # Offset down for max
                         arrowprops=dict(facecolor='black', shrink=0.05),
                         horizontalalignment='center')

        ax2.set_title('R-Precision: ALL Songs (Reconstruction)')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('R-Precision')
        ax2.legend()
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plot 3: R-Precision (HIDDEN)
    if t_r_hidden and v_r_hidden:
        ax3.plot(epochs, t_r_hidden, label='Train R-Prec (Hidden)', color='purple', marker='.')
        ax3.plot(epochs, v_r_hidden, label='Val R-Prec (Hidden)', color='magenta', marker='.')

        # Find best (max)
        if v_r_hidden:
            best_idx = np.argmax(v_r_hidden)
            best_val = v_r_hidden[best_idx]
            best_epoch = epochs[best_idx]

            ax3.scatter(best_epoch, best_val, color='black', s=100, zorder=5)
            text_offset = (max(v_r_hidden) - min(v_r_hidden)) * 0.05 if len(v_r_hidden) > 1 else 0.01
            ax3.annotate(f'Best: {best_val:.4f}',
                         xy=(best_epoch, best_val),
                         xytext=(best_epoch, best_val - text_offset), # Offset down for max
                         arrowprops=dict(facecolor='black', shrink=0.05),
                         horizontalalignment='center')

        ax3.set_title('R-Precision: HIDDEN Songs (Prediction)')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('R-Precision')
        ax3.legend()
        ax3.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    
    # --- SAVE GRAPH ---
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    save_filename = "training_metrics_updated.png"
    save_path = os.path.join(GRAPHS_DIR, save_filename)
    
    plt.savefig(save_path, dpi=300)
    print(f"✅ Graph saved successfully to: {save_path}")

    plt.show()

if __name__ == "__main__":
    plot_history(HISTORY_PATH)