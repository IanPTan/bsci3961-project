import argparse
import tomllib
from pathlib import Path
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from rnn import RNN
from dataset import VAEPathDataset

def get_latest_exp_dir(prefix="rnn_"):
    exp_root = Path("experiments")
    if not exp_root.exists():
        return None
        
    max_num = 0
    latest_dir = None
    for d in exp_root.iterdir():
        if d.is_dir() and d.name.startswith(prefix):
            try:
                num = int(d.name.split("_")[1])
                if num > max_num:
                    max_num = num
                    latest_dir = d
            except ValueError:
                pass
    return latest_dir

@torch.no_grad()
def evaluate_and_save(model, loader, criterion, device, output_h5_path):
    model.eval()
    total_loss = 0.0

    # infer full shapes from one batch
    first_batch = next(iter(loader))
    moves, in_patches, out_patches, _, mask_batch = first_batch

    batch_size0 = moves.shape[0]
    seq_len = moves.shape[1]
    move_dim = moves.shape[2]
    patch_dim = in_patches.shape[2]
    output_dim = out_patches.shape[2]

    num_samples = len(loader.dataset)

    print(f"num_samples: {num_samples}")
    print(f"seq_len:     {seq_len}")
    print(f"move_dim:    {move_dim}")
    print(f"patch_dim:   {patch_dim}")
    print(f"output_dim:  {output_dim}")

    with h5py.File(output_h5_path, "w") as f:
        # internal_loops = 1 (initial) + hidden_loops
        internal_loops = 1 + model.hidden_loops
        
        # preallocate datasets so we can write batch-by-batch
        y_ds = f.create_dataset(
            "y",
            shape=(num_samples, seq_len, output_dim),
            dtype="float32",
        )
        h_ds = f.create_dataset(
            "h_all",
            shape=(num_samples, seq_len, internal_loops, model.hidden_dim),
            dtype="float32",
        )
        mask_ds = f.create_dataset(
            "mask",
            shape=(num_samples, seq_len),
            dtype="float32",
        )

        write_start = 0

        for moves, in_patches, out_patches, _, mask in tqdm(loader, desc="Evaluating val and saving", unit="batch"):
            moves = moves.to(device)
            in_patches = in_patches.to(device)
            out_patches = out_patches.to(device)

            x = torch.cat([in_patches, moves], dim=-1)

            y_pred, _, h_all = model(x, return_all_h=True) # 1: y, 2: h_seq (3D), 3: h_all (4D)
            loss = criterion(y_pred, out_patches)
            total_loss += loss.item()

            bsz = moves.shape[0]
            write_end = write_start + bsz

            # move just this batch to cpu and write immediately
            y_ds[write_start:write_end] = y_pred.detach().cpu().numpy()
            h_ds[write_start:write_end] = h_all.detach().cpu().numpy()
            mask_ds[write_start:write_end] = mask.cpu().numpy()

            write_start = write_end

        # optional metadata
        f.attrs["num_samples"] = num_samples
        f.attrs["seq_len"] = seq_len
        f.attrs["move_dim"] = move_dim
        f.attrs["patch_dim"] = patch_dim
        f.attrs["output_dim"] = output_dim
        f.attrs["hidden_dim"] = model.hidden_dim

    return total_loss / len(loader)

def main():
    parser = argparse.ArgumentParser(description="Evaluate RNN and save hidden states")
    parser.add_argument("--exp-dir", type=str, help="Path to experiment directory")
    args = parser.parse_args()
    
    if args.exp_dir:
        exp_dir = Path(args.exp_dir)
    else:
        exp_dir = get_latest_exp_dir("rnn_")
        if not exp_dir:
            print("Error: No RNN experiment found in experiments/")
            return
            
    print(f"Using experiment directory: {exp_dir}")
    
    config_path = exp_dir / "config.toml"
    if not config_path.exists():
        print(f"Error: {config_path} not found.")
        return
        
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # -------------------------
    # Load validation dataset
    # -------------------------
    val_dataset = VAEPathDataset(
        config["data_path"], 
        split="val",
        mask_prob=0, # No masking for evaluation usually
        mask_start_idx=config["mask_start_idx"]
    )

    sample_moves, sample_in_patches, sample_out_patches, _, _ = val_dataset[0]

    seq_len = sample_in_patches.shape[0]
    patch_dim = sample_in_patches.shape[-1]
    move_dim = sample_moves.shape[-1]
    input_dim = patch_dim + move_dim
    output_dim = sample_out_patches.shape[-1]

    print(f"Sequence length: {seq_len}")
    print(f"Patch dim:       {patch_dim}")
    print(f"Move dim:        {move_dim}")
    print(f"Input dim:       {input_dim}")
    print(f"Output dim:      {output_dim}")
    print(f"Val samples:     {len(val_dataset)}")

    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # -------------------------
    # Load model
    # -------------------------
    model = RNN(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        output_dim=output_dim,
        hidden_loops=config["hidden_loops"],
        k=config["k"],
    ).to(device)

    rnn_weights_path = exp_dir / f"{exp_dir.name}.pt"
    if not rnn_weights_path.exists():
        # Try checkpoint.pt if experiment-named weights don't exist
        rnn_weights_path = exp_dir / "checkpoint.pt"
        if not rnn_weights_path.exists():
            print(f"Error: No weights found in {exp_dir}")
            return
        else:
            print(f"Loading weights from checkpoint: {rnn_weights_path}")
            checkpoint = torch.load(rnn_weights_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"Loading best weights: {rnn_weights_path}")
        model.load_state_dict(torch.load(rnn_weights_path, map_location=device))
        
    model.eval()

    criterion = nn.MSELoss()
    output_h5_path = exp_dir / "val.h5"

    # -------------------------
    # Run val pass and save outputs
    # -------------------------
    val_loss = evaluate_and_save(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        output_h5_path=output_h5_path,
    )

    print(f"Validation loss: {val_loss:.6f}")
    print(f"Saved val predictions and hidden states to {output_h5_path}")

if __name__ == "__main__":
    main()
