import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm
from pytorch_optimizer import OrthoGrad, AdamW, create_optimizer

from rnn import RNN
from dataset import VAEPathDataset


# Hyperparameters
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-3
ADAMW_BETAS = (0.9, 0.999)

# LR Schedule Hyperparameters
WARMUP_PROPORTION = 0.1


# OrthoGrad Hyperparameters
USE_ORTHOGRAD = False
ORTHOGRAD_EPS = 1e-5


HIDDEN_DIM = 4096
HIDDEN_LOOPS = 7
K = 0

# Masking Hyperparameters
MASK_PROB = 0.5      # Probability of masking a patch (0.0 = disabled)
MASK_START_IDX = 4   # Number of initial patches to never mask

MODEL_SAVE_PATH = f"{HIDDEN_DIM}_{HIDDEN_LOOPS}_rnn.pt"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Training...", unit="batch")

    for moves, in_patches, out_patches, _, _ in pbar:
        moves = moves.to(device)
        in_patches = in_patches.to(device)
        out_patches = out_patches.to(device)

        # CONCATENATE PATCH FEATURES + MOVE FEATURES
        x = torch.cat([in_patches, moves], dim=-1)

        optimizer.zero_grad()

        y_pred, h_seq, _ = model(x)
        loss = criterion(y_pred, out_patches)

        loss.backward()
        optimizer.step()
        
        # Step the scheduler (per-batch)
        if scheduler is not None:
            scheduler.step()

        batch_loss = loss.item()
        total_loss += batch_loss
        pbar.set_postfix(mse=f"{batch_loss:.6f}")

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    for moves, in_patches, out_patches, _, _ in tqdm(loader, desc="Evaluating...", unit="batch"):
        moves = moves.to(device)
        in_patches = in_patches.to(device)
        out_patches = out_patches.to(device)

        x = torch.cat([in_patches, moves], dim=-1)

        y_pred, h_seq, _ = model(x)
        loss = criterion(y_pred, out_patches)

        total_loss += loss.item()

    return total_loss / len(loader)


if __name__ == "__main__":
    # LOAD DATASET
    train_dataset = VAEPathDataset(
        "vae_paths.h5", 
        split="train", 
        mask_prob=MASK_PROB, 
        mask_start_idx=MASK_START_IDX
    )
    test_dataset = VAEPathDataset(
        "vae_paths.h5", 
        split="test", 
        mask_prob=MASK_PROB, 
        mask_start_idx=MASK_START_IDX
    )

    # INFER DIMENSIONS FROM ONE SAMPLE
    sample_moves, sample_in_patches, sample_out_patches, _, _ = train_dataset[0]

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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(test_dataset)}")

    # MODEL
    model = RNN(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        output_dim=output_dim,
        hidden_loops=HIDDEN_LOOPS,
        k=K,
    ).to(device)

    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print("loading backup")
    except Exception:
        print("using fresh weights")

    # OPTIMIZER
    optimizer = create_optimizer(
        model,
        optimizer_name='adamw',
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=ADAMW_BETAS,
        use_orthograd=USE_ORTHOGRAD,
        eps=ORTHOGRAD_EPS
    )

    # SCHEDULER
    total_batches = len(train_loader)
    TOTAL_STEPS = NUM_EPOCHS * total_batches
    WARMUP_STEPS = int(TOTAL_STEPS * WARMUP_PROPORTION)
    
    print(f"Total steps:  {TOTAL_STEPS}")
    print(f"Warmup steps: {WARMUP_STEPS}")

    warmup_sch = LinearLR(optimizer, start_factor=0.0001, end_factor=1.0, total_iters=WARMUP_STEPS)
    main_sch = CosineAnnealingLR(optimizer, T_max=(TOTAL_STEPS - WARMUP_STEPS))
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sch, main_sch],
        milestones=[WARMUP_STEPS]
    )

    criterion = nn.MSELoss()

    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Saved best model to {MODEL_SAVE_PATH}")

    print("Training complete.")
    print(f"Best validation loss: {best_val_loss:.6f}")


