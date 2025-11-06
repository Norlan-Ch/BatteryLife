from data_provider.data_loader import Dataset_original, Dataset_AE
from data_provider.data_loader import my_collate_fn_baseline, my_collate_fn_withId, collate_fn_AE, collate_fn_AE_withID
from torch.utils.data import DataLoader, RandomSampler, Dataset

data_dict = {
    'Dataset_original': Dataset_original,
    'Dataset_AE': Dataset_AE
}

def data_provider_baseline_DA(args, flag, tokenizer=None, label_scaler=None, eval_cycle_min=None, eval_cycle_max=None, total_prompts=None, 
                 total_charge_discharge_curves=None, total_curve_attn_masks=None, total_labels=None, unique_labels=None,
                 class_labels=None, life_class_scaler=None, sample_weighted=False, target_dataset='None'):
    Data = data_dict[args.data]

    if flag == 'test' or flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    if flag == 'test' or flag == 'val':
        data_set = Data(args=args,
                flag=flag,
                tokenizer=tokenizer,
                label_scaler=label_scaler,
                eval_cycle_min=eval_cycle_min,
                eval_cycle_max=eval_cycle_max,
                total_prompts=total_prompts, 
                total_charge_discharge_curves=total_charge_discharge_curves, 
                total_curve_attn_masks=total_curve_attn_masks, total_labels=total_labels, unique_labels=unique_labels,
                class_labels=class_labels,
                life_class_scaler=life_class_scaler,
                use_target_dataset=True
            )
    else:
        data_set = Data(args=args,
                flag=flag,
                tokenizer=tokenizer,
                label_scaler=label_scaler,
                eval_cycle_min=eval_cycle_min,
                eval_cycle_max=eval_cycle_max,
                total_prompts=total_prompts, 
                total_charge_discharge_curves=total_charge_discharge_curves, 
                total_curve_attn_masks=total_curve_attn_masks, total_labels=total_labels, unique_labels=unique_labels,
                class_labels=class_labels,
                life_class_scaler=life_class_scaler,
                use_target_dataset=False
            )

    data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last,
                collate_fn=my_collate_fn_baseline)
    
    if target_dataset != 'None' and flag=='train':
        target_data_set = Data(args=args,
                flag=flag,
                tokenizer=tokenizer,
                label_scaler=data_set.return_label_scaler(),
                eval_cycle_min=eval_cycle_min,
                eval_cycle_max=eval_cycle_max,
                total_prompts=total_prompts, 
                total_charge_discharge_curves=total_charge_discharge_curves, 
                total_curve_attn_masks=total_curve_attn_masks, total_labels=total_labels, unique_labels=unique_labels,
                class_labels=class_labels,
                life_class_scaler=data_set.return_life_class_scaler(),
                use_target_dataset=True
            )

        target_data_loader = DataLoader(
                    target_data_set,
                    batch_size=batch_size,
                    shuffle=shuffle_flag,
                    num_workers=args.num_workers,
                    drop_last=drop_last,
                    collate_fn=my_collate_fn_baseline)
        target_sampler = RandomSampler(target_data_loader.dataset, replacement=True, num_samples=len(data_loader.dataset))
        target_resampled_dataloader = DataLoader(target_data_loader.dataset, batch_size=batch_size, sampler=target_sampler, collate_fn=my_collate_fn_baseline)
        return data_set, data_loader, target_data_set, target_resampled_dataloader
    else:
        return data_set, data_loader

def data_provider_baseline(args, flag, tokenizer=None, label_scaler=None, eval_cycle_min=None, eval_cycle_max=None, total_prompts=None, 
                 total_charge_discharge_curves=None, total_curve_attn_masks=None, total_labels=None, unique_labels=None,
                 class_labels=None, life_class_scaler=None, sample_weighted=False):
    Data = data_dict[args.data]

    if flag == 'test' or flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    data_set = Data(args=args,
            flag=flag,
            tokenizer=tokenizer,
            label_scaler=label_scaler,
            eval_cycle_min=eval_cycle_min,
            eval_cycle_max=eval_cycle_max,
            total_prompts=total_prompts, 
            total_charge_discharge_curves=total_charge_discharge_curves, 
            total_curve_attn_masks=total_curve_attn_masks, total_labels=total_labels, unique_labels=unique_labels,
            class_labels=class_labels,
            life_class_scaler=life_class_scaler
        )

    data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last,
                collate_fn=my_collate_fn_baseline)
        
    return data_set, data_loader


def data_provider_evaluate(args, flag, tokenizer=None, label_scaler=None, eval_cycle_min=None, eval_cycle_max=None, total_prompts=None, 
                 total_charge_discharge_curves=None, total_curve_attn_masks=None, total_labels=None, unique_labels=None,
                 class_labels=None, life_class_scaler=None, sample_weighted=False):
    Data = data_dict[args.data]

    if flag == 'test' or flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    data_set = Data(args=args,
            flag=flag,
            tokenizer=tokenizer,
            label_scaler=label_scaler,
            eval_cycle_min=eval_cycle_min,
            eval_cycle_max=eval_cycle_max,
            total_prompts=total_prompts, 
            total_charge_discharge_curves=total_charge_discharge_curves, 
            total_curve_attn_masks=total_curve_attn_masks, total_labels=total_labels, unique_labels=unique_labels,
            class_labels=class_labels,
            life_class_scaler=life_class_scaler
        )


    data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last,
                collate_fn=my_collate_fn_withId) # 与data_provider_baseline的区别
    return data_set, data_loader


def data_provider_AE(args, flag, soh_len=3000, padding_mode='zero',
                     tokenizer=None, label_scaler=None, eval_cycle_min=None, eval_cycle_max=None,
                     total_prompts=None, total_charge_discharge_curves=None, total_curve_attn_masks=None,
                     total_labels=None, unique_labels=None, class_labels=None, life_class_scaler=None,
                     sample_weighted=False):
    """
    Data provider for AutoEncoder in Latent Diffusion Model
    
    Args:
        args: model parameters containing dataset configuration
            - args.dataset: dataset name
            - args.batch_size: batch size
            - args.num_workers: number of workers for DataLoader
        flag: 'train', 'val', or 'test'
        soh_len: target sequence length for padding (None means no padding)
        padding_mode: 'zero' (pad with zeros) or 'last' (pad with last value)
    
    Returns:
        data_set: Dataset_AE instance
        data_loader: DataLoader instance
    
    Example:
        >>> train_set, train_loader = data_provider_AE(args, 'train', soh_len=2000, padding_mode='zero')
        >>> val_set, val_loader = data_provider_AE(args, 'val', soh_len=2000, padding_mode='zero')
        >>> for batch in train_loader:
        >>>     seqs = batch['discharge_capacity_seq']  # [B, soh_len]
        >>>     seq_lengths = batch['seq_length']       # [B]
        >>>     # Training code...
    """

    Data = data_dict[args.data]
    
    # Configure DataLoader settings based on flag
    if flag == 'test' or flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
    
    # Create Dataset_AE instance
    data_set = Data(
        args=args,
        flag=flag,
        soh_len=soh_len,
        padding_mode=padding_mode
    )
    
    # Create DataLoader with custom collate function
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn_AE
    )
    
    return data_set, data_loader


def data_provider_AE_evaluate(args, flag, soh_len=3000, padding_mode='zero',
                              tokenizer=None, label_scaler=None, eval_cycle_min=None, eval_cycle_max=None,
                              total_prompts=None, total_charge_discharge_curves=None, total_curve_attn_masks=None,
                              total_labels=None, unique_labels=None, class_labels=None, life_class_scaler=None,
                              sample_weighted=False):
    """
    Data provider for evaluating AutoEncoder in Latent Diffusion Model
    Similar to data_provider_AE but specifically designed for evaluation
    
    Args:
        args: model parameters containing dataset configuration
            - args.dataset: dataset name
            - args.batch_size: batch size
            - args.num_workers: number of workers for DataLoader
        flag: 'train', 'val', or 'test'
        soh_len: target sequence length for padding (None means no padding)
        padding_mode: 'zero' (pad with zeros) or 'last' (pad with last value)
    
    Returns:
        data_set: Dataset_AE instance
        data_loader: DataLoader instance
    
    Example:
        >>> test_set, test_loader = data_provider_AE_evaluate(args, 'test', soh_len=2000, padding_mode='zero')
        >>> for batch in test_loader:
        >>>     seqs = batch['discharge_capacity_seq']  # [B, soh_len]
        >>>     seq_lengths = batch['seq_length']       # [B]
        >>>     eols = batch['eol']                     # [B]
        >>>     # Evaluation code...
    """

    Data = data_dict[args.data]

    if flag == 'test' or flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
    
    # Create Dataset_AE instance
    data_set = Data(
        args=args,
        flag=flag,
        soh_len=soh_len,
        padding_mode=padding_mode
    )
    
    # Create DataLoader with custom collate function
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn_AE_withID
    )
    
    return data_set, data_loader