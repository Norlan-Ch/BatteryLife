"""
Test script for Dataset_AE with dual outputs (SOH + discharge capacity)
"""
import torch
from data_provider.data_loader import Dataset_AE, collate_fn_AE_withID
from data_provider.data_factory import data_provider_AE
from torch.utils.data import DataLoader

# Test configuration
class Args:
    def __init__(self):
        self.data = 'Dataset_AE'
        self.root_path = "./dataset/"  # 使用定义的路径
        self.dataset = 'exp'
        self.batch_size = 32
        self.num_workers = 0
        self.charge_discharge_length = 100
        self.early_cycle_threshold = 100
        self.seq_len = 5
        self.pred_len = 5
        self.use_multi_gpu = False
        self.devices = '0'
        self.weighted_loss = False  # 添加缺失的属性
        
args = Args()


# Create dataset
print("=" * 80)
print("Testing Dataset_AE with dual outputs (SOH + discharge_capacity)")
print("=" * 80)

train_dataset, dataloader = data_provider_AE(
    args=args,
    flag='train'
)

print(f"\n✓ Dataset created successfully")
print(f"  - Dataset size: {len(train_dataset)}")
print(f"  - Number of SOH sequences: {len(train_dataset.total_soh_seqs)}")
print(f"  - Number of discharge capacity sequences: {len(train_dataset.total_discharge_capacity_seqs)}")

# Test single sample
print("\n" + "=" * 80)
print("Testing single sample")
print("=" * 80)

sample = train_dataset[0]
print(f"\n✓ Sample retrieved successfully")
print(f"  Keys in sample: {list(sample.keys())}")
print(f"  - soh_seq shape: {sample['soh_seq'].shape}")
print(f"  - discharge_capacity_seq shape: {sample['discharge_capacity_seq'].shape}")
print(f"  - eol: {sample['eol']}")
print(f"  - dataset_id: {sample['dataset_id']}")
print(f"  - padding_mask shape: {sample['padding_mask'].shape}")

# Check value ranges
soh_min, soh_max = sample['soh_seq'].min().item(), sample['soh_seq'].max().item()
dc_min, dc_max = sample['discharge_capacity_seq'].min().item(), sample['discharge_capacity_seq'].max().item()
print(f"\n  Value ranges:")
print(f"  - SOH: [{soh_min:.4f}, {soh_max:.4f}] (should be 0-1 range)")
print(f"  - Discharge capacity: [{dc_min:.4f}, {dc_max:.4f}] (in Ah)")

# Test dataloader with collate function
print("\n" + "=" * 80)
print("Testing DataLoader with collate_fn_AE")
print("=" * 80)

dataloader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=collate_fn_AE_withID
)

batch = next(iter(dataloader))
print(f"\n✓ Batch retrieved successfully")
print(f"  Keys in batch: {list(batch.keys())}")
print(f"  - soh_seq shape: {batch['soh_seq'].shape}")
print(f"  - discharge_capacity_seq shape: {batch['discharge_capacity_seq'].shape}")
print(f"  - dataset_id shape: {batch['dataset_id'].shape}")
print(f"  - eol shape: {batch['eol'].shape}")
print(f"  - padding_mask shape: {batch['padding_mask'].shape}")

# Verify relationship between SOH and discharge capacity
print("\n" + "=" * 80)
print("Verifying SOH = discharge_capacity / nominal_capacity relationship")
print("=" * 80)

# Get first sample from batch
soh_sample = batch['soh_seq'][0]
dc_sample = batch['discharge_capacity_seq'][0]
valid_len = batch['eol'][0].item()

# Extract valid portion (non-padded)
soh_valid = soh_sample[:valid_len]
dc_valid = dc_sample[:valid_len]

# Check if SOH is normalized version of discharge capacity
# For a perfect relationship: soh = dc / nominal_capacity
# So: dc_max should approximately equal nominal_capacity when soh ≈ 1
nominal_capacity_estimate = dc_valid.max().item()
soh_from_dc = dc_valid / nominal_capacity_estimate

# Calculate correlation
correlation = torch.corrcoef(torch.stack([soh_valid, soh_from_dc]))[0, 1].item()
print(f"\n✓ Relationship verified")
print(f"  - Estimated nominal capacity: {nominal_capacity_estimate:.4f} Ah")
print(f"  - Correlation between SOH and (DC/nominal): {correlation:.6f}")
print(f"  - Expected: close to 1.0 (perfect correlation)")

if correlation > 0.99:
    print("\n✅ SUCCESS: SOH and discharge capacity have correct relationship!")
else:
    print(f"\n⚠️  WARNING: Correlation {correlation:.4f} is lower than expected")
    print("  This might indicate an issue with the data or normalization")

print("\n" + "=" * 80)
print("Test completed!")
print("=" * 80)
