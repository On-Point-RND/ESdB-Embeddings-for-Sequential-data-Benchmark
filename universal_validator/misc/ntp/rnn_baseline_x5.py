import sys
import os
import glob 
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import pandas as pd
import numpy as np
import dask.dataframe as dd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Next Token Prediction Baseline for Financial Transactions')
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, default='data/x5/', help='Path to data directory')
    parser.add_argument('--max-purchases', type=int, default=50000, help='Maximum number of purchases to use')
    parser.add_argument('--sequence-length', type=int, default=5, help='Sequence length for training')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Train/validation split ratio')
    
    # Model parameters
    parser.add_argument('--rnn-type', type=str, default='gru', choices=['rnn', 'gru', 'lstm'], 
                       help='Type of RNN to use')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--embedding-dim', type=int, default=64, help='Embedding dimension size')
    parser.add_argument('--num-layers', type=int, default=1, help='Number of RNN layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--teacher-forcing-ratio', type=float, default=0.8, 
                       help='Teacher forcing ratio for TF method')
    
    # System parameters
    parser.add_argument('--cuda-devices', type=str, default='2', help='CUDA visible devices')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    
    return parser.parse_args()


class RetailX5Dataset(Dataset):
    def __init__(self, sequence_length=5, data_dir='data/x5/', max_purchases=None):
        self.sequence_length = sequence_length
        self.max_purchases = max_purchases
        
        start_time = time.time()
        
        # Load and preprocess data
        data = {}
        csv_paths = glob.glob(os.path.join(data_dir, '*.csv'))
        csv = ['clients', 'products', 'purchases']
        for csv_path in csv_paths:
            basename = os.path.basename(csv_path)[:-4]
            if basename in csv:
                data[basename] = dd.read_csv(csv_path)
        
        feature_names = "client_id,transaction_datetime,product_id,purchase_sum".split(',')
        all_products = data['products']["product_id,level_4".split(",")]
        
        map_l4 = {}
        for i, r in all_products.iterrows():
            map_l4[r['product_id']] = str(r['level_4'])
        
        purchases = data['purchases'][feature_names]
        purchases['level_4'] = purchases['product_id'].map(map_l4, meta=('level_4', 'string'))
        
        # Apply purchase limit
        if max_purchases:
            self.df = purchases.head(n=max_purchases)
        else:
            self.df = purchases.head(n=len(purchases))
        
        self.df['transaction_datetime'] = pd.to_datetime(self.df['transaction_datetime'])
        
        # Convert datetime to timestamp
        self.df['timestamp'] = self.df['transaction_datetime'].astype('int64') // 10**9
        
        # Fit purchase sum scaler
        self.purchase_scaler = StandardScaler()
        self.purchase_scaler.fit(self.df[['purchase_sum']].values)
        
        load_time = time.time()
        
        # Build vocabulary
        self.vocab = {cat: idx for idx, cat in enumerate(self.df['level_4'].unique())}
        self.vocab_size = len(self.vocab)
        
        # Create sequences and features
        self.sequences = []
        self.targets = []
        self.features = []
        self._create_sequences()
        
        end_time = time.time()
        
        print(f"Dataset created in {end_time - start_time:.2f} seconds")
        print(f"Dataset: {len(self.sequences)} sequences, {self.vocab_size} categories")
        print(f"Purchase stats: mean={self.purchase_scaler.mean_[0]:.2f}, std={self.purchase_scaler.scale_[0]:.2f}")
    
    def _create_sequences(self):
        """Create sequences with time deltas and log1p"""
        self.df['session_id'] = self.df.groupby(['client_id', 'transaction_datetime']).ngroup()
        self.df = self.df.sort_values(['client_id', 'transaction_datetime'])
        
        for client_id in self.df['client_id'].unique():
            client_data = self.df[self.df['client_id'] == client_id]
            sessions = client_data['session_id'].unique()
            
            for i in range(self.sequence_length, len(sessions)):
                context_sessions = sessions[i-self.sequence_length:i]
                target_session = sessions[i]
                
                # Get context items and features
                context_items = []
                context_timestamps = []
                context_purchase_sums = []
                
                for session_id in context_sessions:
                    session_data = client_data[client_data['session_id'] == session_id]
                    # Sort items within session by timestamp
                    session_data = session_data.sort_values('timestamp')
                    for _, row in session_data.iterrows():
                        context_items.append(row['level_4'])
                        context_timestamps.append(row['timestamp'])
                        context_purchase_sums.append(row['purchase_sum'])
                
                # Get target items
                target_data = client_data[client_data['session_id'] == target_session]
                for _, row in target_data.iterrows():
                    if context_items:
                        # Take last sequence_length items
                        seq_items = context_items[-self.sequence_length:]
                        seq_timestamps = context_timestamps[-self.sequence_length:]
                        seq_purchase_sums = context_purchase_sums[-self.sequence_length:]
                        
                        # Calculate time deltas (seconds between consecutive purchases)
                        time_deltas = []
                        for j in range(len(seq_timestamps)):
                            if j == 0:
                                # For first item, use delta from beginning or a small value
                                time_deltas.append(0.0)
                            else:
                                delta_seconds = seq_timestamps[j] - seq_timestamps[j-1]
                                time_deltas.append(delta_seconds)
                        
                        # Apply log1p to time deltas (handles zeros and compresses range)
                        log_time_deltas = np.log1p(time_deltas)  # log(1 + x)
                        
                        # Normalize purchase sums
                        norm_purchase_sums = self.purchase_scaler.transform(
                            np.array(seq_purchase_sums).reshape(-1, 1)
                        ).flatten()
                        
                        # Create feature tensor: [log_time_delta, norm_purchase_sum]
                        features_tensor = torch.tensor(
                            list(zip(log_time_deltas, norm_purchase_sums)), 
                            dtype=torch.float
                        )
                        
                        self.sequences.append(seq_items)
                        self.targets.append(row['level_4'])
                        self.features.append(features_tensor)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        target = self.targets[idx]
        features = self.features[idx]  # Tensor of shape [sequence_length, 2]
        
        seq_indices = [self.vocab[token] for token in sequence]
        
        return (
            torch.tensor(seq_indices, dtype=torch.long),
            torch.tensor(self.vocab[target], dtype=torch.long),
            features
        )


class NextTokenRNN(nn.Module):
    def __init__(self, vocab_size, hidden_dim=128, embedding_dim=64, 
                 continuous_dim=2, rnn_type='gru', num_layers=1, 
                 conditional_dim=0, dropout=0.2):
        super(NextTokenRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.continuous_dim = continuous_dim
        self.conditional_dim = conditional_dim
        self.rnn_type = rnn_type.lower()
        
        # Embedding layer for categorical tokens
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # MLP for continuous features (time_delta, purchase_sum)
        self.continuous_mlp = nn.Sequential(
            nn.Linear(continuous_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32)
        )
        
        # Calculate total input dimension to RNN
        total_input_dim = embedding_dim + 32 + conditional_dim
        
        # RNN layer with drop-in replacement option
        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(total_input_dim, hidden_dim, num_layers, 
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(total_input_dim, hidden_dim, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        else:  # simple RNN
            self.rnn = nn.RNN(total_input_dim, hidden_dim, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Output layer
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, vocab_size)
        )
        
        # Optional conditional embedding projection
        if conditional_dim > 0:
            self.conditional_proj = nn.Linear(conditional_dim, conditional_dim)
        else:
            self.conditional_proj = None
            
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, category_sequence, continuous_features, conditional_embedding=None, hidden=None):
        """
        Args:
            category_sequence: [batch_size, seq_len] - LongTensor of category indices
            continuous_features: [batch_size, seq_len, continuous_dim] - continuous features
            conditional_embedding: [batch_size, conditional_dim] - optional conditional input
            hidden: initial hidden state
        Returns:
            logits: [batch_size, seq_len, vocab_size] - prediction logits
            hidden: final hidden state
        """
        batch_size, seq_len = category_sequence.shape
        
        # 1. Process categorical sequence through embedding
        cat_emb = self.embedding(category_sequence)  # [batch, seq_len, embedding_dim]
        
        # 2. Process continuous features through MLP
        cont_emb = self.continuous_mlp(continuous_features)  # [batch, seq_len, 32]
        
        # 3. Prepare conditional embedding (broadcast across sequence)
        if conditional_embedding is not None and self.conditional_proj is not None:
            cond_emb = self.conditional_proj(conditional_embedding)  # [batch, conditional_dim]
            cond_emb = cond_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, conditional_dim]
        else:
            cond_emb = torch.zeros(batch_size, seq_len, self.conditional_dim, 
                                 device=category_sequence.device)
        
        # 4. Concatenate all inputs
        rnn_input = torch.cat([cat_emb, cont_emb, cond_emb], dim=-1)  # [batch, seq_len, total_dim]
        rnn_input = self.dropout(rnn_input)
        
        # 5. Process through RNN
        rnn_output, hidden = self.rnn(rnn_input, hidden)  # [batch, seq_len, hidden_dim]
        rnn_output = self.dropout(rnn_output)
        
        # 6. Project to vocabulary
        logits = self.output_proj(rnn_output)  # [batch, seq_len, vocab_size]
        
        return logits, hidden
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state"""
        if self.rnn_type == 'lstm':
            h = torch.zeros(self.rnn.num_layers, batch_size, self.hidden_dim, device=device)
            c = torch.zeros(self.rnn.num_layers, batch_size, self.hidden_dim, device=device)
            return (h, c)
        else:
            return torch.zeros(self.rnn.num_layers, batch_size, self.hidden_dim, device=device)


def train_epoch_standard(model, dataloader, optimizer, criterion, device, master_pbar, epoch):
    """Standard next-token prediction with per-batch updates"""
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    total_log_ppl = 0
    
    for batch_idx, (sequences, targets, features) in enumerate(dataloader):
        sequences, features = sequences.to(device), features.to(device)
        
        optimizer.zero_grad()
        
        # Single forward pass
        input_seq = sequences[:, :-1]
        input_feat = features[:, :-1, :]
        target_seq = sequences[:, 1:]
        
        logits, _ = model(input_seq, input_feat)
        loss = criterion(logits.reshape(-1, model.vocab_size), target_seq.reshape(-1))
        loss.backward()
        optimizer.step()
        
        # Metrics
        preds = logits.argmax(dim=-1)
        correct = (preds == target_seq).sum().item()
        
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            target_probs = probs.gather(-1, target_seq.unsqueeze(-1)).squeeze(-1)
            batch_log_ppl = torch.sum(torch.log(target_probs + 1e-8)).item()
        
        total_loss += loss.item()
        total_correct += correct
        total_samples += target_seq.numel()
        total_log_ppl += batch_log_ppl
        
        # Update master progress bar after each batch
        current_acc = correct / target_seq.numel()
        current_ppl = np.exp(-batch_log_ppl / target_seq.numel()) if target_seq.numel() > 0 else 0
        
        master_pbar.set_postfix({
            'Batch': f'{batch_idx+1}/{len(dataloader)}',
            'Loss': f'{loss.item():.2e}',
            'Acc': f'{current_acc:.2e}',
            'PPL': f'{current_ppl:.2e}'
        }, refresh=False)
        master_pbar.update(1)
    
    # Final metrics
    avg_loss = total_loss / len(dataloader)
    final_acc = total_correct / total_samples
    final_ppl = np.exp(-total_log_ppl / total_samples) if total_samples > 0 else 0
    
    return avg_loss, final_acc, final_ppl


def train_epoch_teacher_forcing_simple(model, dataloader, optimizer, criterion, device, master_pbar, epoch, teacher_forcing_ratio=0.8):
    """Simplified teacher forcing with per-batch updates"""
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    total_log_ppl = 0
    
    for batch_idx, (sequences, targets, features) in enumerate(dataloader):
        sequences, features = sequences.to(device), features.to(device)
        batch_size, seq_len = sequences.shape
        
        optimizer.zero_grad()
        
        batch_loss, batch_correct, batch_steps = 0, 0, 0
        batch_log_ppl = 0
        
        # Start with first token
        current_input = sequences[:, 0:1]
        current_features = features[:, 0:1, :]
        hidden = model.init_hidden(batch_size, device)
        
        for t in range(1, seq_len):
            # Forward pass
            logits, hidden = model(current_input, current_features, hidden=hidden)
            last_logits = logits[:, -1, :]
            target_token = sequences[:, t]
            
            # Loss and metrics
            loss = criterion(last_logits, target_token)
            pred = last_logits.argmax(dim=-1)
            correct = (pred == target_token).sum().item()
            
            with torch.no_grad():
                probs = torch.softmax(last_logits, dim=-1)
                target_probs = probs.gather(-1, target_token.unsqueeze(-1)).squeeze(-1)
                batch_log_ppl += torch.sum(torch.log(target_probs + 1e-8)).item()
            
            batch_loss += loss
            batch_correct += correct
            batch_steps += batch_size
            
            # Teacher forcing decision
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            
            if use_teacher_forcing and t < seq_len - 1:
                next_input = sequences[:, t:t+1]
            else:
                next_input = pred.unsqueeze(1)
            
            current_input = next_input
            current_features = features[:, t:t+1, :]
        
        # Backward pass
        avg_batch_loss = batch_loss / (seq_len - 1)
        avg_batch_loss.backward()
        optimizer.step()
        
        total_loss += avg_batch_loss.item()
        total_correct += batch_correct
        total_samples += batch_steps
        total_log_ppl += batch_log_ppl
        
        # Update master progress bar after each batch
        current_acc = batch_correct / batch_steps if batch_steps > 0 else 0
        current_ppl = np.exp(-batch_log_ppl / batch_steps) if batch_steps > 0 else 0
        
        master_pbar.set_postfix({
            'Batch': f'{batch_idx+1}/{len(dataloader)}',
            'Loss': f'{avg_batch_loss.item():.2e}',
            'Acc': f'{current_acc:.2e}',
            'PPL': f'{current_ppl:.2e}'
        }, refresh=False)
        master_pbar.update(1)
    
    # Final metrics
    avg_loss = total_loss / len(dataloader)
    final_acc = total_correct / total_samples
    final_ppl = np.exp(-total_log_ppl / total_samples) if total_samples > 0 else 0
    
    return avg_loss, final_acc, final_ppl


def validate_epoch(model, dataloader, criterion, device):
    """Validation - no progress bar"""
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    total_log_ppl = 0
    
    with torch.no_grad():
        for sequences, targets, features in dataloader:
            sequences, features = sequences.to(device), features.to(device)
            
            input_seq = sequences[:, :-1]
            input_feat = features[:, :-1, :]
            target_seq = sequences[:, 1:]
            
            logits, _ = model(input_seq, input_feat)
            loss = criterion(logits.reshape(-1, model.vocab_size), target_seq.reshape(-1))
            preds = logits.argmax(dim=-1)
            correct = (preds == target_seq).sum().item()
            
            probs = torch.softmax(logits, dim=-1)
            target_probs = probs.gather(-1, target_seq.unsqueeze(-1)).squeeze(-1)
            batch_log_ppl = torch.sum(torch.log(target_probs + 1e-8)).item()
            
            total_loss += loss.item()
            total_correct += correct
            total_samples += target_seq.numel()
            total_log_ppl += batch_log_ppl
    
    avg_loss = total_loss / len(dataloader)
    final_acc = total_correct / total_samples
    final_ppl = np.exp(-total_log_ppl / total_samples) if total_samples > 0 else 0
    
    return avg_loss, final_acc, final_ppl


def weight_reset(m):
    """Reset model weights"""
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()


def compare_training_methods(model, train_loader, val_loader, optimizer, criterion, device, epochs=5, teacher_forcing_ratio=0.8):
    """
    Compare standard next-token prediction vs simplified teacher forcing
    """
    methods = {
        'standard': train_epoch_standard,
        'teacher_forcing': lambda model, dataloader, optimizer, criterion, device, master_pbar, epoch: 
            train_epoch_teacher_forcing_simple(model, dataloader, optimizer, criterion, device, master_pbar, epoch, teacher_forcing_ratio)
    }
    
    results = defaultdict(list)
    
    for method_name, train_func in methods.items():
        print(f"\n{'='*80}")
        print(f"TRAINING WITH: {method_name.upper()} METHOD")
        print(f"{'='*80}")
        
        # Reset model and optimizer for fair comparison
        model.apply(weight_reset)
        optimizer = optim.Adam(model.parameters(), lr=optimizer.param_groups[0]['lr'])
        
        method_results = {
            'train_loss': [], 'train_acc': [], 'train_ppl': [],
            'val_loss': [], 'val_acc': [], 'val_ppl': [],
            'epoch_times': []
        }
        
        # Calculate total iterations for more frequent updates
        total_iterations = epochs * len(train_loader)
        
        # Create master progress bar for all iterations - minimal bar, maximum postfix
        master_pbar = tqdm(total=total_iterations, desc=f'{method_name.upper():<8}', 
                  position=0, leave=False, 
                  bar_format='{desc} {percentage:1.0f}%|{bar:5}| {n_fmt}/{total_fmt} {postfix}',
                  ncols=120)
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training with per-batch updates
            train_loss, train_acc, train_ppl = train_func(model, train_loader, optimizer, 
                                                         criterion, device, master_pbar, epoch)
            
            # Validation
            val_loss, val_acc, val_ppl = validate_epoch(model, val_loader, criterion, device)
            
            epoch_time = time.time() - epoch_start
            
            # Store results
            method_results['train_loss'].append(train_loss)
            method_results['train_acc'].append(train_acc)
            method_results['train_ppl'].append(train_ppl)
            method_results['val_loss'].append(val_loss)
            method_results['val_acc'].append(val_acc)
            method_results['val_ppl'].append(val_ppl)
            method_results['epoch_times'].append(epoch_time)
            
            # Calculate moving averages (last 3 epochs)
            window = min(3, epoch + 1)
            mov_avg_train_loss = np.mean(method_results['train_loss'][-window:])
            mov_avg_train_acc = np.mean(method_results['train_acc'][-window:])
            mov_avg_train_ppl = np.mean(method_results['train_ppl'][-window:])
            mov_avg_val_loss = np.mean(method_results['val_loss'][-window:])
            mov_avg_val_acc = np.mean(method_results['val_acc'][-window:])
            mov_avg_val_ppl = np.mean(method_results['val_ppl'][-window:])
            
            # Update master progress bar with epoch summary
            master_pbar.set_postfix({
                'Epoch': f'{epoch+1}/{epochs}',
                'Trn_L': f'{mov_avg_train_loss:.2e}',
                'Trn_A': f'{mov_avg_train_acc:.2e}', 
                'Trn_P': f'{mov_avg_train_ppl:.2e}',
                'Val_L': f'{mov_avg_val_loss:.2e}',
                'Val_A': f'{mov_avg_val_acc:.2e}',
                'Val_P': f'{mov_avg_val_ppl:.2e}',
                'Time': f'{epoch_time:.1f}s'
            }, refresh=False)
        
        master_pbar.close()
        results[method_name] = method_results
        
        # Print final epoch results for this method
        print(f"\nFINAL RESULTS - {method_name.upper()}:")
        print(f"Train Loss: {train_loss:.2e}, Acc: {train_acc:.2e}, PPL: {train_ppl:.2e}")
        print(f"Val Loss:   {val_loss:.2e}, Acc: {val_acc:.2e}, PPL: {val_ppl:.2e}")
        print(f"Avg Epoch Time: {np.mean(method_results['epoch_times']):.1f}s")
    
    return results


def print_comparison_report(results):
    """Print comparison table with proper f-string formatting"""
    print(f"\n{'='*100}")
    print("FINAL COMPARISON REPORT: STANDARD vs TEACHER FORCING")
    print(f"{'='*100}")
    
    headers = ["Method", "Train Acc", "Val Acc", "Train PPL", "Val PPL", "Best Val Acc", "Avg Time"]
    print(f"{headers[0]:<15} {headers[1]:<12} {headers[2]:<12} {headers[3]:<12} {headers[4]:<12} {headers[5]:<12} {headers[6]:<10}")
    print("-" * 90)
    
    for method_name, method_results in results.items():
        final_train_acc = method_results['train_acc'][-1]
        final_val_acc = method_results['val_acc'][-1]
        final_train_ppl = method_results['train_ppl'][-1]
        final_val_ppl = method_results['val_ppl'][-1]
        best_val_acc = max(method_results['val_acc'])
        avg_epoch_time = np.mean(method_results['epoch_times'])
        
        print(f"{method_name:<15} {final_train_acc:12.2f} {final_val_acc:12.2f} "
              f"{final_train_ppl:12.2f} {final_val_ppl:12.2f} "
              f"{best_val_acc:12.2f} {avg_epoch_time:10.1f}")


def run_comparison_experiment(dataset, args, device):
    # Split dataset
    dataset_size = len(dataset)
    train_size = int(args.train_ratio * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = NextTokenRNN(
        vocab_size=dataset.vocab_size,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        continuous_dim=2,
        rnn_type=args.rnn_type,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Model: {args.rnn_type.upper()}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Dataset: {len(dataset)} sequences")
    print(f"Train/Val: {len(train_dataset)}/{len(val_dataset)}")
    
    # Run comparison
    results = compare_training_methods(
        model, train_loader, val_loader, optimizer, criterion, device, 
        epochs=args.epochs, teacher_forcing_ratio=args.teacher_forcing_ratio
    )
    
    return results


def main():
    args = parse_args()
    
    # Set CUDA devices
    if not args.no_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    
    print("=" * 80)
    print("NEXT TOKEN PREDICTION BASELINE FOR FINANCIAL TRANSACTIONS")
    print("=" * 80)
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("=" * 80)
    
    # Create dataset
    print("Creating dataset...")
    dataset_start = time.time()
    dataset = RetailX5Dataset(
        sequence_length=args.sequence_length,
        data_dir=args.data_dir,
        max_purchases=args.max_purchases
    )
    dataset_end = time.time()
    print(f"Total dataset creation time: {dataset_end - dataset_start:.2f} seconds\n")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Run the experiment
    print("Starting comparison experiment...")
    results = run_comparison_experiment(dataset, args, device)
    
    # Print report
    print_comparison_report(results)


if __name__ == "__main__":
    main()
