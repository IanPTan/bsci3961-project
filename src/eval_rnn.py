import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from rnn import RNN
from dataset import VAEPathDataset


# -------------------------
# Config
# -------------------------
PATHS_H5 = "vae_paths.h5"

BATCH_SIZE = 32

# must match the trained model
HIDDEN_DIM = 2048
HIDDEN_LOOPS = 9
K = 0

RNN_WEIGHTS = f"{HIDDEN_DIM}_{HIDDEN_LOOPS}_rnn.pt"
OUTPUT_H5 = f"{HIDDEN_DIM}_{HIDDEN_LOOPS}_val.h5"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


@torch.no_grad()
def evaluate_and_save(model, loader, criterion, device, output_h5_path):
    model.eval()
    total_loss = 0.0

    # infer full shapes from one batch
    first_batch = next(iter(loader))
    moves, in_patches, out_patches, _ = first_batch

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

        write_start = 0

        for moves, in_patches, out_patches, _ in tqdm(loader, desc="Evaluating val and saving", unit="batch"):
            moves = moves.to(device)
            in_patches = in_patches.to(device)
            out_patches = out_patches.to(device)

            x = torch.cat([in_patches, moves], dim=-1)

            y_pred, _, h_all = model(x) # 1: y, 2: h_seq (3D), 3: h_all (4D)
            loss = criterion(y_pred, out_patches)
            total_loss += loss.item()

            bsz = moves.shape[0]
            write_end = write_start + bsz

            # move just this batch to cpu and write immediately
            y_ds[write_start:write_end] = y_pred.detach().cpu().numpy()
            h_ds[write_start:write_end] = h_all.detach().cpu().numpy()

            write_start = write_end

        # optional metadata
        f.attrs["num_samples"] = num_samples
        f.attrs["seq_len"] = seq_len
        f.attrs["move_dim"] = move_dim
        f.attrs["patch_dim"] = patch_dim
        f.attrs["output_dim"] = output_dim
        f.attrs["hidden_dim"] = model.hidden_dim

    return total_loss / len(loader)


if __name__ == "__main__":
    # -------------------------
    # Load validation dataset
    # -------------------------
    val_dataset = VAEPathDataset(PATHS_H5, split="val")

    sample_moves, sample_in_patches, sample_out_patches, _ = val_dataset[0]

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

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # -------------------------
    # Load model
    # -------------------------
    model = RNN(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        output_dim=output_dim,
        hidden_loops=HIDDEN_LOOPS,
        k=K,
    ).to(device)

    model.load_state_dict(torch.load(RNN_WEIGHTS, map_location=device))
    model.eval()
    print(f"Loaded model from {RNN_WEIGHTS}")

    criterion = nn.MSELoss()

    # -------------------------
    # Run val pass and save outputs
    # -------------------------
    val_loss = evaluate_and_save(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        output_h5_path=OUTPUT_H5,
    )

    print(f"Validation loss: {val_loss:.6f}")
    print(f"Saved val predictions and hidden states to {OUTPUT_H5}")
