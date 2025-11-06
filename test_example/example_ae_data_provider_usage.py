"""
Example usage of data_provider_AE and data_provider_AE_evaluate
for Latent Diffusion Model AutoEncoder training and evaluation

This script demonstrates how to use the data provider functions 
to easily create datasets and dataloaders for AE training.
"""

import torch
import torch.nn as nn
import argparse
from data_provider.data_factory import data_provider_AE, data_provider_AE_evaluate


def create_args(batch_size=4):  # 修改默认batch_size
    """Create mock arguments for data provider"""
    args = argparse.Namespace()
    args.root_path = './dataset'  # Update this to your dataset path
    args.dataset = 'exp'          # Choose from available datasets
    args.data = 'Dataset_AE'
    args.batch_size = batch_size  # 使用参数
    args.num_workers = 0  # 改为0避免多进程问题
    return args


def example_basic_usage():
    """Example 1: Basic usage with data_provider_AE"""
    print("=" * 80)
    print("Example 1: Basic usage with data_provider_AE")
    print("=" * 80)
    
    args = create_args(batch_size=2)  # 使用小的batch_size适应小数据集
    
    # Create train, val, test datasets and dataloaders
    train_set, train_loader = data_provider_AE(
        args=args,
        flag='train',
        soh_len=2000,
        padding_mode='zero'
    )
    
    val_set, val_loader = data_provider_AE(
        args=args,
        flag='val',
        soh_len=2000,
        padding_mode='zero'
    )
    
    test_set, test_loader = data_provider_AE(
        args=args,
        flag='test',
        soh_len=2000,
        padding_mode='zero'
    )
    
    print(f"Dataset sizes:")
    print(f"  Train: {len(train_set)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_set)} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(test_set)} samples, {len(test_loader)} batches")
    
    # Get a batch
    if len(train_loader) > 0:
        batch = next(iter(train_loader))
        print(f"\nBatch information:")
        print(f"  soh_seq shape: {batch['soh_seq'].shape}")
        print(f"  discharge_capacity_seq shape: {batch['discharge_capacity_seq'].shape}")
        print(f"  seq_length shape: {batch['eol'].shape}")
        print(f"  eol shape: {batch['eol'].shape}")
        print(f"  mask shape: {batch['padding_mask'].shape}")
    else:
        print(f"\n⚠ Warning: No batches available (dataset too small or batch_size too large)")


def example_evaluation_usage():
    """Example 2: Using data_provider_AE_evaluate for evaluation"""
    print("\n" + "=" * 80)
    print("Example 2: Using data_provider_AE_evaluate for evaluation")
    print("=" * 80)
    
    args = create_args(batch_size=2)
    
    # Create test dataset for evaluation (no shuffling, no drop_last)
    test_set, test_loader = data_provider_AE_evaluate(
        args=args,
        flag='test',
        soh_len=2000,
        padding_mode='zero'
    )
    
    print(f"Evaluation dataset:")
    print(f"  Test: {len(test_set)} samples, {len(test_loader)} batches")
    print(f"  Note: Evaluation uses shuffle=False and drop_last=False")
    
    # Simulate evaluation loop
    if len(test_loader) > 0:
        print(f"\nSimulating evaluation:")
        total_samples = 0
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= 3:  # Only show first 3 batches
                print(f"  ... (total {len(test_loader)} batches)")
                break
            
            seqs = batch['discharge_capacity_seq']
            seq_lengths = batch['eol']
            eols = batch['eol']
            
            total_samples += seqs.shape[0]
            print(f"  Batch {batch_idx}: {seqs.shape[0]} samples, "
                  f"seq_lengths range: [{seq_lengths.min().item()}, {seq_lengths.max().item()}]")


def example_different_padding_modes():
    """Example 3: Comparing different padding modes"""
    print("\n" + "=" * 80)
    print("Example 3: Comparing different padding modes")
    print("=" * 80)
    
    args = create_args(batch_size=2)
    
    # Zero padding
    train_set_zero, train_loader_zero = data_provider_AE(
        args=args,
        flag='train',
        soh_len=2000,
        padding_mode='zero'
    )
    
    # Last value padding
    train_set_last, train_loader_last = data_provider_AE(
        args=args,
        flag='train',
        soh_len=2000,
        padding_mode='last'
    )
    
    if len(train_loader_zero) > 0 and len(train_loader_last) > 0:
        # Get batches
        batch_zero = next(iter(train_loader_zero))
        batch_last = next(iter(train_loader_last))
        
        # Compare padding
        idx = 0
        seq_zero = batch_zero['discharge_capacity_seq'][idx]
        seq_last = batch_last['discharge_capacity_seq'][idx]
        seq_len = batch_zero['eol'][idx].item()
        
        print(f"\nSample {idx} comparison (seq_length={seq_len}):")
        print(f"  Zero padding - padded region: {seq_zero[seq_len:seq_len+5].tolist()}")
        print(f"  Last padding - padded region: {seq_last[seq_len:seq_len+5].tolist()}")
        print(f"  Last valid value: {seq_zero[seq_len-1].item():.4f}")


def example_no_padding():
    """Example 4: Using without padding (variable length sequences)"""
    print("\n" + "=" * 80)
    print("Example 4: Using without padding (variable length sequences)")
    print("=" * 80)
    
    args = create_args(batch_size=1)  # Must use batch_size=1 for variable length
    
    # No padding
    train_set, train_loader = data_provider_AE(
        args=args,
        flag='train',
        soh_len=None,  # No padding
        padding_mode='zero'
    )
    
    print(f"Dataset without padding:")
    print(f"  Train: {len(train_set)} samples")
    print(f"  Note: Using batch_size=1 because sequences have different lengths")
    
    # Show first 3 samples
    if len(train_loader) > 0:
        print(f"\nFirst 3 samples:")
        for i, batch in enumerate(train_loader):
            if i >= 3:
                break
            seq = batch['discharge_capacity_seq'][0]
            seq_len = batch['eol'][0].item()
            print(f"  Sample {i}: seq_length={seq_len}, shape={seq.shape}")


def example_complete_ae_training():
    """Example 5: Complete AutoEncoder training structure"""
    print("\n" + "=" * 80)
    print("Example 5: Complete AutoEncoder training structure")
    print("=" * 80)
    
    args = create_args(batch_size=2)
    
    # Hyperparameters
    soh_len = 2000
    latent_dim = 128
    num_epochs = 2  # 减少epoch数
    
    print(f"Training configuration:")
    print(f"  SOH length: {soh_len}")
    print(f"  Latent dimension: {latent_dim}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Number of epochs: {num_epochs}")
    
    # Create datasets
    train_set, train_loader = data_provider_AE(args, 'train', soh_len=soh_len, padding_mode='zero')
    val_set, val_loader = data_provider_AE(args, 'val', soh_len=soh_len, padding_mode='zero')
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_set)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_set)} samples, {len(val_loader)} batches")
    
    if len(train_loader) == 0:
        print(f"\n⚠ Warning: No training batches available, skipping training example")
        return
    
    # Simple AutoEncoder (for demonstration)
    class SimpleAE(nn.Module):
        def __init__(self, input_dim, latent_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, latent_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, input_dim)
            )
        
        def forward(self, x):
            z = self.encoder(x)
            x_recon = self.decoder(z)
            return x_recon, z
    
    # Initialize model, optimizer, criterion
    model = SimpleAE(input_dim=soh_len, latent_dim=latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss(reduction='none')  # Use 'none' to apply mask
    
    print(f"\nModel architecture:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop structure
    print(f"\nTraining loop structure:")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 2:  # Only show first 2 batches
                if len(train_loader) > 2:
                    print(f"    ... (total {len(train_loader)} batches)")
                break
            
            # Extract data - 使用discharge_capacity_seq
            seqs = batch['discharge_capacity_seq']  # [B, soh_len]
            seq_lengths = batch['eol']       # [B]
            mask = batch['padding_mask']                    # [B, soh_len]
            
            # Forward pass
            x_recon, z = model(seqs)
            
            # Compute loss with mask (only on valid positions)
            loss = criterion(x_recon, seqs)  # [B, L]
            loss = (loss * mask).sum() / mask.sum()  # Masked average
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            print(f"  Epoch {epoch}, Batch {batch_idx}: loss={loss.item():.6f}")
        
        # Validation
        if epoch == 0:
            print(f"  (Validation loop similar to training, but without gradient updates)")
    
    print(f"\nKey points:")
    print(f"  1. Batch contains 'padding_mask' key for masking padded regions")
    print(f"  2. Apply mask in loss: (loss * mask).sum() / mask.sum()")
    print(f"  3. Mask shape: [B, soh_len], same as sequence shape")
    print(f"  4. This ensures padding regions don't affect training")


def example_multiple_datasets():
    """Example 6: Loading multiple datasets for comparison"""
    print("\n" + "=" * 80)
    print("Example 6: Loading multiple datasets for comparison")
    print("=" * 80)
    
    dataset_names = ['exp', 'MATR', 'HUST', 'CALCE']
    soh_len = 2000
    
    for dataset_name in dataset_names:
        try:
            args = create_args(batch_size=2)
            args.dataset = dataset_name
            
            train_set, train_loader = data_provider_AE(args, 'train', soh_len=soh_len)
            val_set, val_loader = data_provider_AE(args, 'val', soh_len=soh_len)
            test_set, test_loader = data_provider_AE(args, 'test', soh_len=soh_len)
            
            print(f"\n{dataset_name}:")
            print(f"  Train: {len(train_set):4d} samples, {len(train_loader):3d} batches")
            print(f"  Val:   {len(val_set):4d} samples, {len(val_loader):3d} batches")
            print(f"  Test:  {len(test_set):4d} samples, {len(test_loader):3d} batches")
            
            # Get sample statistics
            if len(train_set) > 0:
                sample = train_set[0]
                print(f"  Sample EOL: {sample['eol']}")
                print(f"  Discharge capacity shape: {sample['discharge_capacity_seq'].shape}")
        except Exception as e:
            print(f"\n{dataset_name}: Failed to load ({str(e)})")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Data Provider Usage Examples for AutoEncoder")
    print("=" * 80)
    
    try:
        example_basic_usage()
        example_evaluation_usage()
        example_different_padding_modes()
        example_complete_ae_training()
        example_multiple_datasets()
    except FileNotFoundError as e:
        print(f"\nError: Dataset not found. Please update 'root_path' in create_args().")
        print(f"Current error: {e}")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease make sure:")
        print("  1. Dataset path is correctly set")
        print("  2. Dataset files are in the correct location")
        print("  3. Required dependencies are installed")
    
    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)
    print("\nQuick reference:")
    print("  - Training:   data_provider_AE(args, 'train', soh_len=2000, padding_mode='zero')")
    print("  - Validation: data_provider_AE(args, 'val', soh_len=2000, padding_mode='zero')")
    print("  - Evaluation: data_provider_AE_evaluate(args, 'test', soh_len=2000, padding_mode='zero')")
    print("\nImportant notes:")
    print("  - Batch contains 'soh_seq', 'discharge_capacity_seq', 'padding_mask', 'eol', 'eol'")
    print("  - Use 'padding_mask' to ignore padded regions in loss computation")
    print("  - For small datasets, use smaller batch_size (e.g., 2-4)")
    print("=" * 80)
    