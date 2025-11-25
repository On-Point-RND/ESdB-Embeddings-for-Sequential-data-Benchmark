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
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import dask.dataframe as dd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Next Token Prediction Baseline for AGE Financial Transactions')
    
    # Data parameters
    parser.add_argument('--data-url', type=str, 
                       default='https://huggingface.co/datasets/dllllb/age-group-prediction/resolve/main/transactions_train.csv.gz?download=true',
                       help='URL to download AGE dataset')
    parser.add_argument('--data-dir', type=str, default='./age_data/', help='Path to cache data directory')
    parser.add_argument('-mx', '--max-transactions', type=int, default=50000, help='Maximum number of transactions to use')
    parser.add_argument('--sequence-length', type=int, default=5, help='Sequence length for training')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Train/validation split ratio')
    parser.add_argument('--split-method', type=str, default='strict_temporal', 
                       choices=['temporal', 'client', 'strict_temporal'], 
                       help='How to split data: temporal (by time), client (by client), strict_temporal (global time split)')
    parser.add_argument('--cross-session', action='store_true', help='Create sequences across sessions')
    parser.add_argument('--min-frequency', type=int, default=10, help='Minimum frequency for vocabulary items')
    
    # Model parameters
    parser.add_argument('-rnn', '--rnn-type', type=str, default='lstm', choices=['rnn', 'gru', 'lstm'], 
                       help='Type of RNN to use')
    parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--embedding-dim', type=int, default=64, help='Embedding dimension size')
    parser.add_argument('--num-layers', type=int, default=1, help='Number of RNN layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--teacher-forcing-ratio', type=float, default=0.5,
                       help='Teacher forcing ratio for TF method')
    
    # System parameters
    parser.add_argument('--cuda-devices', type=str, default='0', help='CUDA visible devices')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    
    return parser.parse_args()


class AGEDataset(Dataset):
    def __init__(self, sequence_length=5, data_url=None, data_dir='./age_data/', max_transactions=None, 
                 split_type='all', split_ratio=0.8, split_method='strict_temporal', 
                 cross_session=False, scaler=None, vocab=None, min_frequency=10, global_time_split=None):
        self.sequence_length = sequence_length
        self.max_transactions = max_transactions
        self.split_type = split_type
        self.split_ratio = split_ratio
        self.split_method = split_method
        self.cross_session = cross_session
        self.min_frequency = min_frequency
        
        start_time = time.time()
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Download and load data
        cache_file = os.path.join(data_dir, 'transactions_train.csv')
        if not os.path.exists(cache_file):
            print("Downloading AGE dataset...")
            self.df = pd.read_csv(data_url, compression='gzip')
            self.df.to_csv(cache_file, index=False)
        else:
            print("Loading cached AGE dataset...")
            self.df = pd.read_csv(cache_file)
        
        # Apply transaction limit
        if max_transactions and len(self.df) > max_transactions:
            self.df = self.df.head(max_transactions)
        
        # Convert trans_date to datetime (treat as sequential days)
        base_date = pd.Timestamp('2023-01-01')
        self.df['transaction_datetime'] = base_date + pd.to_timedelta(self.df['trans_date'] - 1, unit='D')
        
        # Sort by client and date
        self.df = self.df.sort_values(['client_id', 'transaction_datetime'])
        
        # Convert datetime to timestamp
        self.df['timestamp'] = self.df['transaction_datetime'].astype('int64') // 10**9
        
        # Apply appropriate split method
        self.df = self._apply_split(self.df, global_time_split)
        
        # Use provided scaler or fit new one
        if scaler is not None:
            self.amount_scaler = scaler
        else:
            # Fit scaler on amount_rur for training data
            self.amount_scaler = StandardScaler()
            if len(self.df) > 0:
                self.amount_scaler.fit(self.df[['amount_rur']].values)
        
        # Use provided vocab or create new one
        if vocab is not None:
            self.vocab = vocab
            self.is_using_training_vocab = True
            if '<OOV>' not in self.vocab:
                self.vocab['<OOV>'] = len(self.vocab)
        else:
            # Create vocabulary with frequency filtering
            self.vocab = self._create_vocab_with_frequency()
            self.is_using_training_vocab = False
        
        self.vocab_size = len(self.vocab)
        self.oov_token = '<OOV>'
        
        print(f"Vocabulary size: {self.vocab_size} (after frequency filtering)")
        
        # Create sequences and features
        self.sequences = []
        self.targets = []
        self.features = []
        self.sequence_timestamps = []
        
        if self.cross_session:
            self._create_sequences_cross_session()
        else:
            self._create_sequences_within_session()
        
        end_time = time.time()
        
        print(f"Dataset created in {end_time - start_time:.2f} seconds")
        print(f"Dataset: {len(self.sequences)} sequences, {self.vocab_size} small_group categories")
        if len(self.df) > 0:
            print(f"Time range: {self.df['transaction_datetime'].min()} to {self.df['transaction_datetime'].max()}")
        if scaler is None and len(self.df) > 0:
            print(f"Amount stats: mean={self.amount_scaler.mean_[0]:.2f}, std={self.amount_scaler.scale_[0]:.2f}")
    
    def _create_vocab_with_frequency(self):
        """Create vocabulary with frequency filtering to reduce rare categories"""
        counter = Counter(self.df['small_group'].astype(str))
        
        # Keep only items that meet minimum frequency
        vocab = {}
        idx = 0
        
        # Add special tokens first
        vocab['<OOV>'] = idx
        idx += 1
        
        # Add frequent items
        for item, count in counter.most_common():
            if count >= self.min_frequency:
                vocab[str(item)] = idx
                idx += 1
        
        print(f"Vocabulary: {len(vocab)} items (filtered from {len(counter)}), min_freq={self.min_frequency}")
        print(f"Most common items: {list(counter.most_common(10))}")
        
        return vocab
    
    def _apply_split(self, df, global_time_split=None):
        """Apply the appropriate split method to the dataframe"""
        if self.split_type == 'all':
            return df
        
        if self.split_method == 'strict_temporal':
            return self._apply_strict_temporal_split(df, global_time_split)
        elif self.split_method == 'temporal':
            return self._apply_temporal_split_fixed(df)
        elif self.split_method == 'client':
            return self._apply_client_split(df)
        else:
            raise ValueError(f"Unknown split method: {self.split_method}")
    
    def _apply_strict_temporal_split(self, df, global_time_split=None):
        """Strict temporal split - all clients use the same time threshold"""
        if len(df) == 0:
            return df
            
        if global_time_split is None:
            # Calculate global split time
            sorted_df = df.sort_values('transaction_datetime')
            split_idx = int(len(sorted_df) * self.split_ratio)
            global_time_split = sorted_df.iloc[split_idx]['transaction_datetime']
            print(f"Global time split at: {global_time_split}")
        
        if self.split_type == 'train':
            result = df[df['transaction_datetime'] < global_time_split]
        else:  # 'val'
            result = df[df['transaction_datetime'] >= global_time_split]
        
        print(f"{self.split_type.upper()} split: {len(result)} transactions from {result['client_id'].nunique()} clients")
        return result
    
    def _apply_temporal_split_fixed(self, df):
        """Split by time per client to avoid leakage"""
        if len(df) == 0:
            return df
            
        train_dfs = []
        val_dfs = []
        
        for client_id in df['client_id'].unique():
            client_data = df[df['client_id'] == client_id].sort_values('transaction_datetime')
            if len(client_data) == 0:
                continue
                
            split_idx = int(len(client_data) * self.split_ratio)
            
            if self.split_type == 'train':
                train_dfs.append(client_data.iloc[:split_idx])
            else:  # 'val'
                if split_idx < len(client_data):  # Only add if client has validation data
                    val_dfs.append(client_data.iloc[split_idx:])
        
        if self.split_type == 'train':
            result = pd.concat(train_dfs) if train_dfs else df.iloc[0:0]
        else:
            result = pd.concat(val_dfs) if val_dfs else df.iloc[0:0]
        
        print(f"{self.split_type.upper()} split: {len(result)} transactions from {result['client_id'].nunique()} clients")
        return result
    
    def _apply_client_split(self, df):
        """Split by client only - unseen clients in validation"""
        if len(df) == 0:
            return df
            
        unique_clients = df['client_id'].unique()
        split_idx = int(len(unique_clients) * self.split_ratio)
        
        train_clients = unique_clients[:split_idx]
        val_clients = unique_clients[split_idx:]
        
        if self.split_type == 'train':
            return df[df['client_id'].isin(train_clients)]
        else:  # 'val'
            return df[df['client_id'].isin(val_clients)]
    
    def _create_sequences_within_session(self):
        """Create sequences ONLY within the same shopping session"""
        if len(self.df) == 0:
            return
            
        # For AGE dataset, we'll define sessions by client_id and transaction_datetime
        self.df['session_id'] = self.df.groupby(['client_id', 'transaction_datetime']).ngroup()
        self.df = self.df.sort_values(['client_id', 'transaction_datetime', 'timestamp'])
        
        sequences_created = 0
        sequences_skipped_oov = 0
        
        for client_id in self.df['client_id'].unique():
            client_data = self.df[self.df['client_id'] == client_id]
            
            for session_id in client_data['session_id'].unique():
                session_data = client_data[client_data['session_id'] == session_id]
                
                # Sort items within session by timestamp
                session_data = session_data.sort_values('timestamp')
                
                # Get all items in this session
                session_items = session_data['small_group'].astype(str).tolist()
                session_timestamps = session_data['timestamp'].tolist()
                session_amounts = session_data['amount_rur'].tolist()
                
                # Only create sequences if session has enough items
                if len(session_items) > self.sequence_length:
                    for i in range(self.sequence_length, len(session_items)):
                        # Input: previous sequence_length items in this session
                        seq_items = session_items[i-self.sequence_length:i]
                        seq_timestamps = session_timestamps[i-self.sequence_length:i]
                        seq_amounts = session_amounts[i-self.sequence_length:i]
                        
                        # Target: next item in this same session
                        target_item = session_items[i]
                        
                        # Skip sequences with OOV targets that aren't in training vocab
                        if self.is_using_training_vocab:
                            if any(item not in self.vocab for item in seq_items) or target_item not in self.vocab:
                                sequences_skipped_oov += 1
                                continue
                        
                        # Calculate time deltas WITHIN the same session
                        time_deltas = []
                        for j in range(len(seq_timestamps)):
                            if j == 0:
                                time_deltas.append(0.0)  # First item in sequence
                            else:
                                delta_seconds = seq_timestamps[j] - seq_timestamps[j-1]
                                time_deltas.append(delta_seconds)
                        
                        # Apply log1p to time deltas
                        log_time_deltas = np.log1p(time_deltas)
                        
                        # Normalize amounts using the fitted scaler
                        norm_amounts = self.amount_scaler.transform(
                            np.array(seq_amounts).reshape(-1, 1)
                        ).flatten()
                        
                        # Create feature tensor with both time deltas and normalized amounts
                        features_tensor = torch.tensor(
                            list(zip(log_time_deltas, norm_amounts)), 
                            dtype=torch.float
                        )
                        
                        self.sequences.append(seq_items)
                        self.targets.append(target_item)
                        self.features.append(features_tensor)
                        self.sequence_timestamps.append(session_timestamps[i])
                        sequences_created += 1
        
        print(f"Created {sequences_created} sequences, skipped {sequences_skipped_oov} due to OOV")
    
    def _create_sequences_cross_session(self):
        """Create sequences across sessions for more training data"""
        if len(self.df) == 0:
            return
            
        self.df = self.df.sort_values(['client_id', 'timestamp'])
        
        sequences_created = 0
        sequences_skipped_oov = 0
        
        for client_id in self.df['client_id'].unique():
            client_data = self.df[self.df['client_id'] == client_id]
            
            # Get all items for this client across all sessions
            client_items = client_data['small_group'].astype(str).tolist()
            client_timestamps = client_data['timestamp'].tolist()
            client_amounts = client_data['amount_rur'].tolist()
            
            # Create sequences across sessions
            if len(client_items) > self.sequence_length:
                for i in range(self.sequence_length, len(client_items)):
                    seq_items = client_items[i-self.sequence_length:i]
                    target_item = client_items[i]
                    
                    # Skip OOV targets in validation
                    if self.is_using_training_vocab:
                        if any(item not in self.vocab for item in seq_items) or target_item not in self.vocab:
                            sequences_skipped_oov += 1
                            continue
                    
                    # Calculate time deltas (can be across sessions)
                    seq_timestamps = client_timestamps[i-self.sequence_length:i]
                    time_deltas = [0.0]  # First item
                    for j in range(1, len(seq_timestamps)):
                        delta_seconds = seq_timestamps[j] - seq_timestamps[j-1]
                        time_deltas.append(delta_seconds)
                    
                    log_time_deltas = np.log1p(time_deltas)
                    
                    # Normalize amounts
                    seq_amounts = client_amounts[i-self.sequence_length:i]
                    norm_amounts = self.amount_scaler.transform(
                        np.array(seq_amounts).reshape(-1, 1)
                    ).flatten()
                    
                    features_tensor = torch.tensor(
                        list(zip(log_time_deltas, norm_amounts)), 
                        dtype=torch.float
                    )
                    
                    self.sequences.append(seq_items)
                    self.targets.append(target_item)
                    self.features.append(features_tensor)
                    self.sequence_timestamps.append(client_timestamps[i])
                    sequences_created += 1
        
        print(f"Created {sequences_created} sequences, skipped {sequences_skipped_oov} due to OOV")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        target = self.targets[idx]
        features = self.features[idx]
        
        # Handle OOV tokens - use <OOV> for items not in vocabulary
        seq_indices = [self.vocab.get(token, self.vocab[self.oov_token]) for token in sequence]
        target_index = self.vocab.get(target, self.vocab[self.oov_token])
        
        return (
            torch.tensor(seq_indices, dtype=torch.long),
            torch.tensor(target_index, dtype=torch.long),
            features
        )


class NextTokenRNN(nn.Module):
    def __init__(self, vocab_size, hidden_dim=64, embedding_dim=64, 
                 continuous_dim=2, rnn_type='lstm', num_layers=1, 
                 conditional_dim=0, dropout=0.5):
        super(NextTokenRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.continuous_dim = continuous_dim
        self.conditional_dim = conditional_dim
        self.rnn_type = rnn_type.lower()
        
        # Embedding layer for categorical tokens
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Simpler MLP for continuous features
        self.continuous_mlp = nn.Sequential(
            nn.Linear(continuous_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16)
        )
        
        # Calculate total input dimension to RNN
        total_input_dim = embedding_dim + 16 + conditional_dim
        
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
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, category_sequence, continuous_features, conditional_embedding=None, hidden=None):
        batch_size, seq_len = category_sequence.shape
        
        # 1. Process categorical sequence through embedding
        cat_emb = self.embedding(category_sequence)
        
        # 2. Process continuous features through MLP
        cont_emb = self.continuous_mlp(continuous_features)
        
        # 3. Prepare conditional embedding (broadcast across sequence)
        if conditional_embedding is not None:
            cond_emb = conditional_embedding.unsqueeze(1).expand(-1, seq_len, -1)
        else:
            cond_emb = torch.zeros(batch_size, seq_len, self.conditional_dim, 
                                 device=category_sequence.device)
        
        # 4. Concatenate all inputs
        rnn_input = torch.cat([cat_emb, cont_emb, cond_emb], dim=-1)
        rnn_input = self.dropout(rnn_input)
        
        # 5. Process through RNN
        rnn_output, hidden = self.rnn(rnn_input, hidden)
        rnn_output = self.dropout(rnn_output)
        
        # 6. Project to vocabulary
        logits = self.output_proj(rnn_output)
        
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
    """Standard next-token prediction with gradient clipping"""
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    total_log_ppl = 0
    
    for batch_idx, (sequences, targets, features) in enumerate(dataloader):
        sequences, targets, features = sequences.to(device), targets.to(device), features.to(device)
        
        optimizer.zero_grad()
        
        logits, _ = model(sequences, features)
        last_logits = logits[:, -1, :]
        
        loss = criterion(last_logits, targets)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        # Metrics
        preds = last_logits.argmax(dim=-1)
        correct = (preds == targets).sum().item()
        
        with torch.no_grad():
            probs = torch.softmax(last_logits, dim=-1)
            target_probs = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            batch_log_ppl = torch.sum(torch.log(target_probs + 1e-8)).item()
        
        total_loss += loss.item()
        total_correct += correct
        total_samples += targets.numel()
        total_log_ppl += batch_log_ppl
        
        # Update progress bar
        current_acc = correct / targets.numel()
        current_ppl = np.exp(-batch_log_ppl / targets.numel()) if targets.numel() > 0 else 0
        
        master_pbar.set_postfix({
            'Batch': f'{batch_idx+1}/{len(dataloader)}',
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{current_acc:.4f}',
            'PPL': f'{current_ppl:.4f}'
        }, refresh=False)
        master_pbar.update(1)
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    final_acc = total_correct / total_samples if total_samples > 0 else 0
    final_ppl = np.exp(-total_log_ppl / total_samples) if total_samples > 0 else 0
    
    return avg_loss, final_acc, final_ppl


def train_epoch_teacher_forcing_simple(model, dataloader, optimizer, criterion, device, master_pbar, epoch, teacher_forcing_ratio=0.5):
    """Simplified teacher forcing"""
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    total_log_ppl = 0
    
    for batch_idx, (sequences, targets, features) in enumerate(dataloader):
        sequences, targets, features = sequences.to(device), targets.to(device), features.to(device)
        batch_size = sequences.shape[0]
        
        optimizer.zero_grad()
        
        batch_loss, batch_correct, batch_steps = 0, 0, 0
        batch_log_ppl = 0
        
        # Start with empty sequence, build up to predict target
        current_input = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
        current_features = torch.zeros(batch_size, 0, 2, device=device)
        hidden = model.init_hidden(batch_size, device)
        
        # Build context step by step
        for t in range(sequences.shape[1]):
            # Add next context item
            next_context_item = sequences[:, t:t+1]
            next_context_features = features[:, t:t+1, :]
            
            # Concatenate to current input
            current_input = torch.cat([current_input, next_context_item], dim=1)
            current_features = torch.cat([current_features, next_context_features], dim=1)
            
            # Forward pass with current context
            logits, hidden = model(current_input, current_features, hidden=hidden)
            last_logits = logits[:, -1, :]
            
            # If we have full context, predict the actual target
            if t == sequences.shape[1] - 1:
                target_token = targets
                
                # Loss and metrics for final prediction
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
        
        # Backward pass
        if batch_loss > 0:
            avg_batch_loss = batch_loss
            avg_batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
        
        total_loss += avg_batch_loss.item() if batch_loss > 0 else 0
        total_correct += batch_correct
        total_samples += batch_steps
        total_log_ppl += batch_log_ppl
        
        # Update progress bar
        current_acc = batch_correct / batch_steps if batch_steps > 0 else 0
        current_ppl = np.exp(-batch_log_ppl / batch_steps) if batch_steps > 0 else 0
        
        master_pbar.set_postfix({
            'Batch': f'{batch_idx+1}/{len(dataloader)}',
            'Loss': f'{avg_batch_loss.item() if batch_loss > 0 else 0:.4f}',
            'Acc': f'{current_acc:.4f}',
            'PPL': f'{current_ppl:.4f}'
        }, refresh=False)
        master_pbar.update(1)
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    final_acc = total_correct / total_samples if total_samples > 0 else 0
    final_ppl = np.exp(-total_log_ppl / total_samples) if total_samples > 0 else 0
    
    return avg_loss, final_acc, final_ppl


def validate_epoch(model, dataloader, criterion, device):
    """Validation"""
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    total_log_ppl = 0
    
    with torch.no_grad():
        for sequences, targets, features in dataloader:
            sequences, targets, features = sequences.to(device), targets.to(device), features.to(device)
            
            logits, _ = model(sequences, features)
            last_logits = logits[:, -1, :]
            
            loss = criterion(last_logits, targets)
            preds = last_logits.argmax(dim=-1)
            correct = (preds == targets).sum().item()
            
            # Perplexity calculation
            probs = torch.softmax(last_logits, dim=-1)
            target_probs = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            batch_log_ppl = torch.sum(torch.log(target_probs + 1e-8)).item()
            
            total_loss += loss.item()
            total_correct += correct
            total_samples += targets.numel()
            total_log_ppl += batch_log_ppl
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    final_acc = total_correct / total_samples if total_samples > 0 else 0
    final_ppl = np.exp(-total_log_ppl / total_samples) if total_samples > 0 else 0
    
    return avg_loss, final_acc, final_ppl


def weight_reset(m):
    """Reset model weights"""
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()


def compare_training_methods(model, train_loader, val_loader, optimizer, criterion, device, epochs=100, teacher_forcing_ratio=0.5):
    """
    Compare standard next-token prediction vs simplified teacher forcing
    """
    methods = {
        'standard': train_epoch_standard,
        'teacher_forcing': lambda model, dataloader, optimizer, criterion, device, master_pbar, epoch: 
            train_epoch_teacher_forcing_simple(model, dataloader, optimizer, criterion, device, master_pbar, epoch, teacher_forcing_ratio)
    }
    
    results = defaultdict(list)
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=20, factor=0.5,
    )
    
    for method_name, train_func in methods.items():
        print(f"\n{'='*80}")
        print(f"TRAINING WITH: {method_name.upper()} METHOD")
        print(f"{'='*80}")
        
        # Reset model and optimizer for fair comparison
        model.apply(weight_reset)
        optimizer = optim.AdamW(model.parameters(), lr=optimizer.param_groups[0]['lr'], weight_decay=1e-4)
        
        method_results = {
            'train_loss': [], 'train_acc': [], 'train_ppl': [],
            'val_loss': [], 'val_acc': [], 'val_ppl': [],
            'epoch_times': [], 'learning_rates': []
        }
        
        # Calculate total iterations for more frequent updates
        total_iterations = epochs * len(train_loader)
        
        # Create master progress bar for all iterations
        master_pbar = tqdm(total=total_iterations, desc=f'{method_name.upper():<8}', 
                  position=0, leave=False, 
                  bar_format='{desc} {percentage:1.0f}%|{bar:5}| {n_fmt}/{total_fmt} {postfix}',
                  ncols=120)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 100
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training with per-batch updates
            train_loss, train_acc, train_ppl = train_func(model, train_loader, optimizer, 
                                                         criterion, device, master_pbar, epoch)
            
            # Validation
            val_loss, val_acc, val_ppl = validate_epoch(model, val_loader, criterion, device)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            epoch_time = time.time() - epoch_start
            
            # Store results
            method_results['train_loss'].append(train_loss)
            method_results['train_acc'].append(train_acc)
            method_results['train_ppl'].append(train_ppl)
            method_results['val_loss'].append(val_loss)
            method_results['val_acc'].append(val_acc)
            method_results['val_ppl'].append(val_ppl)
            method_results['epoch_times'].append(epoch_time)
            method_results['learning_rates'].append(current_lr)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
            
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
                'Trn_L': f'{mov_avg_train_loss:.4f}',
                'Trn_A': f'{mov_avg_train_acc:.4f}', 
                'Trn_P': f'{mov_avg_train_ppl:.4f}',
                'Val_L': f'{mov_avg_val_loss:.4f}',
                'Val_A': f'{mov_avg_val_acc:.4f}',
                'Val_P': f'{mov_avg_val_ppl:.4f}',
                'LR': f'{current_lr:.2e}',
                'Time': f'{epoch_time:.1f}s'
            }, refresh=False)
        
        master_pbar.close()
        results[method_name] = method_results
        
        # Print final epoch results for this method
        print(f"\nFINAL RESULTS - {method_name.upper()}:")
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, PPL: {train_ppl:.4f}")
        print(f"Val Loss:   {val_loss:.4f}, Acc: {val_acc:.4f}, PPL: {val_ppl:.4f}")
        print(f"Best Val Loss: {best_val_loss:.4f}")
        print(f"Avg Epoch Time: {np.mean(method_results['epoch_times']):.1f}s")
    
    return results


def print_comparison_report(results):
    """Print comparison table with proper f-string formatting"""
    print(f"\n{'='*100}")
    print("FINAL COMPARISON REPORT")
    print(f"{'='*100}")
    
    headers = ["Method", "Train Acc", "Val Acc", "Train PPL", "Val PPL", "Best Val Acc", "Best Val Loss", "Avg Time"]
    print(f"{headers[0]:<15} {headers[1]:<12} {headers[2]:<12} {headers[3]:<12} {headers[4]:<12} {headers[5]:<12} {headers[6]:<12} {headers[7]:<10}")
    print("-" * 110)
    
    for method_name, method_results in results.items():
        final_train_acc = method_results['train_acc'][-1]
        final_val_acc = method_results['val_acc'][-1]
        final_train_ppl = method_results['train_ppl'][-1]
        final_val_ppl = method_results['val_ppl'][-1]
        best_val_acc = max(method_results['val_acc'])
        best_val_loss = min(method_results['val_loss'])
        avg_epoch_time = np.mean(method_results['epoch_times'])
        
        print(f"{method_name:<15} {final_train_acc:12.4f} {final_val_acc:12.4f} "
              f"{final_train_ppl:12.4f} {final_val_ppl:12.4f} "
              f"{best_val_acc:12.4f} {best_val_loss:12.4f} {avg_epoch_time:10.1f}")


def run_comparison_experiment(args, device):
    # Calculate global time split for strict temporal splitting
    if args.split_method == 'strict_temporal':
        # Load full data to calculate global split time
        temp_dataset = AGEDataset(
            sequence_length=args.sequence_length,
            data_url=args.data_url,
            data_dir=args.data_dir,
            max_transactions=args.max_transactions,
            split_type='all',
            split_ratio=args.train_ratio,
            split_method='strict_temporal'
        )
        global_time_split = temp_dataset.df.sort_values('transaction_datetime').iloc[
            int(len(temp_dataset.df) * args.train_ratio)
        ]['transaction_datetime']
        print(f"Using global time split: {global_time_split}")
    else:
        global_time_split = None

    # Create training dataset
    print("Creating training dataset...")
    train_dataset = AGEDataset(
        sequence_length=args.sequence_length,
        data_url=args.data_url,
        data_dir=args.data_dir,
        max_transactions=args.max_transactions,
        split_type='train',
        split_ratio=args.train_ratio,
        split_method=args.split_method,
        cross_session=args.cross_session,
        min_frequency=args.min_frequency,
        global_time_split=global_time_split
    )
    
    # Create validation dataset using training vocab and scaler
    print("Creating validation dataset...")
    val_dataset = AGEDataset(
        sequence_length=args.sequence_length,
        data_url=args.data_url,
        data_dir=args.data_dir,
        max_transactions=args.max_transactions,
        split_type='val',
        split_ratio=args.train_ratio,
        split_method=args.split_method,
        cross_session=args.cross_session,
        scaler=train_dataset.amount_scaler,
        vocab=train_dataset.vocab,
        min_frequency=args.min_frequency,
        global_time_split=global_time_split
    )
    
    print(f"Training sequences: {len(train_dataset)}")
    print(f"Validation sequences: {len(val_dataset)}")
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("ERROR: One of the datasets is empty!")
        return {}
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model with smaller capacity
    model = NextTokenRNN(
        vocab_size=train_dataset.vocab_size,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        continuous_dim=2,
        rnn_type=args.rnn_type,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    # Use AdamW with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Model: {args.rnn_type.upper()}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Run comparison
    results = compare_training_methods(
        model, train_loader, val_loader, optimizer, criterion, device, 
        epochs=args.epochs, teacher_forcing_ratio=args.teacher_forcing_ratio
    )
    
    return results


def main():
    args = parse_args()
    
    if not args.no_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    
    print("=" * 80)
    print("NEXT TOKEN PREDICTION BASELINE FOR AGE FINANCIAL TRANSACTIONS")
    print("=" * 80)
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("=" * 80)
        
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    print("Starting comparison experiment...")
    results = run_comparison_experiment(args, device)
    
    if results:
        print_comparison_report(results)


if __name__ == "__main__":
    main()
