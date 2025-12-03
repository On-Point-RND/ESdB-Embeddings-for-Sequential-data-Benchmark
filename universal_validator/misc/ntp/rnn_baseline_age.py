import sys
import os
import glob 
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description='Full Autoregressive Next Token Prediction for AGE Financial Transactions')
    
    # Data parameters
    parser.add_argument('--data-url', type=str, 
                       default='https://huggingface.co/datasets/dllllb/age-group-prediction/resolve/main/transactions_train.csv.gz?download=true',
                       help='URL to download AGE dataset')
    parser.add_argument('--data-dir', type=str, default='./age_data/', help='Path to cache data directory')
    parser.add_argument('-mx', '--max-transactions', type=int, default=None, help='Maximum number of transactions to use')
    parser.add_argument('--max-seq-length', type=int, default=10, help='Maximum sequence length')
    parser.add_argument('--min-seq-length', type=int, default=2, help='Minimum sequence length to include')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Train/validation split ratio')
    parser.add_argument('--min-frequency', type=int, default=10, help='Minimum frequency for vocabulary items')
    
    # Model parameters
    parser.add_argument('-rnn', '--rnn_type', type=str, default='lstm', choices=['rnn', 'gru', 'lstm'], 
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
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('--teacher-forcing-ratio', type=float, default=0.5,
                       help='Teacher forcing ratio (used for teacher forcing method)')
    
    # System parameters
    parser.add_argument('--cuda-devices', type=str, default='0', help='CUDA visible devices')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    
    return parser.parse_args()


class AGEDataset(Dataset):
    def __init__(self, max_seq_length=10, min_seq_length=2, data_url=None, data_dir='./age_data/', 
                 max_transactions=None, split_type='train', split_ratio=0.8, 
                 scaler=None, vocab=None, min_frequency=10):
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.max_transactions = max_transactions
        self.split_type = split_type
        self.split_ratio = split_ratio
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
        
        # Apply simple temporal split per client
        self.df = self._apply_simple_split(self.df)
        
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
            # Ensure special tokens exist
            if '<PAD>' not in self.vocab:
                self.vocab['<PAD>'] = len(self.vocab)
            if '<OOV>' not in self.vocab:
                self.vocab['<OOV>'] = len(self.vocab)
            if '<START>' not in self.vocab:
                self.vocab['<START>'] = len(self.vocab)
            if '<END>' not in self.vocab:
                self.vocab['<END>'] = len(self.vocab)
        else:
            # Create vocabulary with frequency filtering and special tokens
            self.vocab = self._create_vocab_with_frequency()
            self.is_using_training_vocab = False
        
        self.vocab_size = len(self.vocab)
        
        print(f"Vocabulary size: {self.vocab_size} (after frequency filtering)")
        print(f"Special tokens: { {k: self.vocab[k] for k in ['<PAD>', '<OOV>', '<START>', '<END>'] if k in self.vocab} }")
        
        # Create sequences - variable length
        self.sequences = []
        self.sequence_targets = []  # Full target sequences
        self.features = []
        self.sequence_lengths = []
        
        self._create_variable_length_sequences()
        
        end_time = time.time()
        
        print(f"Dataset created in {end_time - start_time:.2f} seconds")
        print(f"Dataset: {len(self.sequences)} sequences, vocabulary size: {self.vocab_size}")
        print(f"Sequence length stats: min={min(self.sequence_lengths) if self.sequence_lengths else 0}, "
              f"max={max(self.sequence_lengths) if self.sequence_lengths else 0}, "
              f"avg={np.mean(self.sequence_lengths) if self.sequence_lengths else 0:.1f}")
        if len(self.df) > 0:
            print(f"Time range: {self.df['transaction_datetime'].min()} to {self.df['transaction_datetime'].max()}")
    
    def _create_vocab_with_frequency(self):
        """Create vocabulary with frequency filtering to reduce rare categories"""
        counter = Counter(self.df['small_group'].astype(str))
        
        # Create vocabulary with special tokens first
        vocab = {}
        idx = 0
        
        # Add special tokens
        vocab['<PAD>'] = idx; idx += 1
        vocab['<OOV>'] = idx; idx += 1
        vocab['<START>'] = idx; idx += 1
        vocab['<END>'] = idx; idx += 1
        
        # Add frequent items
        for item, count in counter.most_common():
            if count >= self.min_frequency:
                vocab[str(item)] = idx
                idx += 1
        
        print(f"Vocabulary: {len(vocab)} items (filtered from {len(counter)}), min_freq={self.min_frequency}")
        print(f"Most common items: {list(counter.most_common(5))}")
        
        return vocab
    
    def _apply_simple_split(self, df):
        """Simple temporal split per client"""
        if self.split_type == 'all':
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
    
    def _create_variable_length_sequences(self):
        """Create variable length sequences for each client"""
        if len(self.df) == 0:
            return
            
        sequences_created = 0
        sequences_skipped_oov = 0
        sequences_skipped_short = 0
        
        for client_id in self.df['client_id'].unique():
            client_data = self.df[self.df['client_id'] == client_id].sort_values('timestamp')
            
            if len(client_data) < self.min_seq_length:
                sequences_skipped_short += 1
                continue
            
            # Get all items for this client in chronological order
            client_items = client_data['small_group'].astype(str).tolist()
            client_timestamps = client_data['timestamp'].tolist()
            client_amounts = client_data['amount_rur'].tolist()
            
            # Create sequences of increasing length
            for seq_len in range(self.min_seq_length, min(len(client_items), self.max_seq_length) + 1):
                seq_items = client_items[:seq_len]
                seq_timestamps = client_timestamps[:seq_len]
                seq_amounts = client_amounts[:seq_len]
                
                # Skip sequences with OOV tokens that aren't in training vocab
                if self.is_using_training_vocab:
                    if any(item not in self.vocab for item in seq_items):
                        sequences_skipped_oov += 1
                        continue
                
                # Calculate time deltas between consecutive transactions
                time_deltas = []
                for j in range(len(seq_timestamps)):
                    if j == 0:
                        time_deltas.append(0.0)  # First item in sequence
                    else:
                        delta_seconds = seq_timestamps[j] - seq_timestamps[j-1]
                        time_deltas.append(delta_seconds)
                
                log_time_deltas = np.log1p(time_deltas)
                
                # Normalize amounts using the fitted scaler
                if len(seq_amounts) > 0:
                    norm_amounts = self.amount_scaler.transform(
                        np.array(seq_amounts).reshape(-1, 1)
                    ).flatten()
                else:
                    norm_amounts = np.array([])
                
                # Create feature tensor
                features_tensor = torch.tensor(
                    list(zip(log_time_deltas, norm_amounts)), 
                    dtype=torch.float
                )
                
                self.sequences.append(seq_items)
                self.features.append(features_tensor)
                self.sequence_lengths.append(seq_len)
                sequences_created += 1
        
        print(f"Created {sequences_created} sequences, "
              f"skipped {sequences_skipped_oov} due to OOV, "
              f"skipped {sequences_skipped_short} due to being too short")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        features = self.features[idx]
        seq_len = self.sequence_lengths[idx]
        
        # Convert tokens to indices
        seq_indices = [self.vocab.get(token, self.vocab['<OOV>']) for token in sequence]
        
        # Input: <START> + sequence[:-1]
        # Target: sequence (shifted by 1)
        input_indices = [self.vocab['<START>']] + seq_indices[:-1]
        target_indices = seq_indices
        
        return (
            torch.tensor(input_indices, dtype=torch.long),
            torch.tensor(target_indices, dtype=torch.long),
            features,
            seq_len
        )


class NextTokenRNN(nn.Module):
    def __init__(self, vocab_size, hidden_dim=64, embedding_dim=64, 
                 continuous_dim=2, rnn_type='lstm', num_layers=1, 
                 dropout=0.5):
        super(NextTokenRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.continuous_dim = continuous_dim
        self.rnn_type = rnn_type.lower()
        
        # Embedding layer for categorical tokens
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # MLP for continuous features
        self.continuous_mlp = nn.Sequential(
            nn.Linear(continuous_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16)
        )
        
        # Calculate total input dimension to RNN
        total_input_dim = embedding_dim + 16
        
        # RNN layer
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
    
    def forward(self, category_sequence, continuous_features, lengths=None, hidden=None):
        batch_size, seq_len = category_sequence.shape
        
        # 1. Process categorical sequence through embedding
        cat_emb = self.embedding(category_sequence)
        
        # 2. Process continuous features through MLP
        cont_emb = self.continuous_mlp(continuous_features)
        
        # 3. Concatenate all inputs
        rnn_input = torch.cat([cat_emb, cont_emb], dim=-1)
        rnn_input = self.dropout(rnn_input)
        
        # 4. Pack sequences if lengths are provided
        if lengths is not None:
            # Move lengths to CPU for pack_padded_sequence
            lengths_cpu = lengths.cpu()
            rnn_input = pack_padded_sequence(rnn_input, lengths_cpu, batch_first=True, enforce_sorted=False)
        
        # 5. Process through RNN
        rnn_output, hidden = self.rnn(rnn_input, hidden)
        
        # 6. Unpack if packed
        if lengths is not None:
            rnn_output, _ = pad_packed_sequence(rnn_output, batch_first=True)
        
        rnn_output = self.dropout(rnn_output)
        
        # 7. Project to vocabulary
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


def collate_fn(batch):
    """Custom collate function to handle variable length sequences"""
    inputs, targets, features, lengths = zip(*batch)
    
    # Pad sequences
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)  # 0 = <PAD>
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    features_padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    
    # Convert lengths to tensor
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    
    # Sort by length (descending) for packed sequences
    lengths_sorted, sorted_idx = lengths_tensor.sort(descending=True)
    inputs_sorted = inputs_padded[sorted_idx]
    targets_sorted = targets_padded[sorted_idx]
    features_sorted = features_padded[sorted_idx]
    
    return inputs_sorted, targets_sorted, features_sorted, lengths_sorted


def train_epoch_standard(model, dataloader, optimizer, criterion, device, master_pbar, epoch):
    """Standard full autoregressive training (predict all tokens)"""
    model.train()
    total_loss, total_tokens_correct, total_tokens = 0, 0, 0
    total_sequences_correct, total_sequences = 0, 0
    total_log_ppl = 0
    
    for batch_idx, (inputs, targets, features, lengths) in enumerate(dataloader):
        inputs, targets, features = inputs.to(device), targets.to(device), features.to(device)
        lengths = lengths.to(device)
        
        batch_size, seq_len = inputs.shape
        
        optimizer.zero_grad()
        
        # Forward pass through the entire sequence
        logits, _ = model(inputs, features, lengths.cpu())
        
        # Calculate loss (ignore padding tokens)
        loss = 0
        batch_tokens_correct = 0
        batch_tokens = 0
        batch_sequences_correct = 0
        
        # We need to calculate loss per position, ignoring padding
        for i in range(batch_size):
            seq_len_i = lengths[i].item()
            # Only consider non-padding positions
            seq_logits = logits[i, :seq_len_i]  # [seq_len_i, vocab_size]
            seq_targets = targets[i, :seq_len_i]  # [seq_len_i]
            
            seq_loss = criterion(seq_logits, seq_targets)
            loss += seq_loss
            
            # Token-level accuracy
            seq_preds = seq_logits.argmax(dim=-1)
            seq_correct = (seq_preds == seq_targets).sum().item()
            batch_tokens_correct += seq_correct
            batch_tokens += seq_len_i
            
            # Sequence-level accuracy (exact match)
            if (seq_preds == seq_targets).all().item():
                batch_sequences_correct += 1
            
            # Perplexity
            with torch.no_grad():
                probs = torch.softmax(seq_logits, dim=-1)
                target_probs = probs.gather(-1, seq_targets.unsqueeze(-1)).squeeze(-1)
                total_log_ppl += torch.sum(torch.log(target_probs + 1e-8)).item()
        
        loss = loss / batch_size  # Average over sequences
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        total_tokens_correct += batch_tokens_correct
        total_tokens += batch_tokens
        total_sequences_correct += batch_sequences_correct
        total_sequences += batch_size
        
        # Update progress bar
        token_acc = batch_tokens_correct / batch_tokens if batch_tokens > 0 else 0
        seq_acc = batch_sequences_correct / batch_size if batch_size > 0 else 0
        current_ppl = np.exp(-total_log_ppl / total_tokens) if total_tokens > 0 else 0
        
        master_pbar.set_postfix({
            'Batch': f'{batch_idx+1}/{len(dataloader)}',
            'Loss': f'{loss.item():.2f}',
            'TokAcc': f'{token_acc:.2f}',
            'SeqAcc': f'{seq_acc:.2f}',
            'PPL': f'{current_ppl:.2f}'
        }, refresh=False)
        master_pbar.update(1)
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    token_accuracy = total_tokens_correct / total_tokens if total_tokens > 0 else 0
    sequence_accuracy = total_sequences_correct / total_sequences if total_sequences > 0 else 0
    perplexity = np.exp(-total_log_ppl / total_tokens) if total_tokens > 0 else 0
    
    return avg_loss, token_accuracy, sequence_accuracy, perplexity


def train_epoch_teacher_forcing(model, dataloader, optimizer, criterion, device, master_pbar, epoch, teacher_forcing_ratio=0.5):
    """Teacher forcing training for full sequence prediction"""
    model.train()
    total_loss, total_tokens_correct, total_tokens = 0, 0, 0
    total_sequences_correct, total_sequences = 0, 0
    total_log_ppl = 0
    
    for batch_idx, (inputs, targets, features, lengths) in enumerate(dataloader):
        inputs, targets, features = inputs.to(device), targets.to(device), features.to(device)
        lengths = lengths.to(device)
        
        batch_size, max_seq_len = inputs.shape
        
        optimizer.zero_grad()
        
        batch_loss = 0
        batch_tokens_correct = 0
        batch_tokens = 0
        batch_sequences_correct = 0
        
        # Initialize hidden state
        hidden = model.init_hidden(batch_size, device)
        
        # Start with <START> token for all sequences
        current_input = inputs[:, :1]  # [batch_size, 1] - just the <START> token
        current_features = features[:, :1, :]  # [batch_size, 1, 2]
        
        # Store predictions for each position
        all_predictions = []
        
        # Autoregressive prediction for each position
        for t in range(max_seq_len):
            # Get actual length for each sequence
            active_mask = (t < lengths).unsqueeze(1)  # [batch_size, 1]
            
            # Forward pass
            logits, hidden = model(current_input, current_features, hidden=hidden)
            last_logits = logits[:, -1, :]  # [batch_size, vocab_size]
            
            # Get target for this position
            target_token = targets[:, t] if t < max_seq_len else torch.zeros(batch_size, dtype=torch.long, device=device)
            
            # Calculate loss only for active sequences
            active_loss = criterion(last_logits, target_token)
            # Mask loss for sequences that have ended
            loss = (active_loss * active_mask.squeeze()).sum() / active_mask.sum().clamp(min=1)
            batch_loss += loss
            
            # Get predictions
            pred = last_logits.argmax(dim=-1)  # [batch_size]
            all_predictions.append(pred.unsqueeze(1))  # [batch_size, 1]
            
            # Token-level accuracy (only for active sequences)
            active_correct = ((pred == target_token) & active_mask.squeeze()).sum().item()
            batch_tokens_correct += active_correct
            batch_tokens += active_mask.sum().item()
            
            # Perplexity for active sequences
            with torch.no_grad():
                probs = torch.softmax(last_logits, dim=-1)
                target_probs = probs.gather(-1, target_token.unsqueeze(-1)).squeeze(-1)
                # Only for active sequences
                active_log_ppl = torch.sum(torch.log(target_probs + 1e-8) * active_mask.squeeze())
                total_log_ppl += active_log_ppl.item()
            
            # Teacher forcing: decide next input
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            
            if use_teacher_forcing:
                # Use ground truth as next input
                next_input = target_token.unsqueeze(1)  # [batch_size, 1]
            else:
                # Use model's own prediction as next input
                next_input = pred.unsqueeze(1)  # [batch_size, 1]
            
            # Get features for next position if available
            if t + 1 < max_seq_len:
                next_features = features[:, t+1:t+2, :]
            else:
                # Use zeros for positions beyond sequence length
                next_features = torch.zeros(batch_size, 1, 2, device=device)
            
            # Update current input
            current_input = torch.cat([current_input, next_input], dim=1)
            current_features = torch.cat([current_features, next_features], dim=1)
        
        # Backward pass
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        # Calculate sequence-level accuracy
        if all_predictions:
            predictions = torch.cat(all_predictions, dim=1)  # [batch_size, max_seq_len]
            for i in range(batch_size):
                seq_len_i = lengths[i].item()
                seq_preds = predictions[i, :seq_len_i]
                seq_targets = targets[i, :seq_len_i]
                if (seq_preds == seq_targets).all().item():
                    batch_sequences_correct += 1
        
        # Update metrics
        total_loss += batch_loss.item()
        total_tokens_correct += batch_tokens_correct
        total_tokens += batch_tokens
        total_sequences_correct += batch_sequences_correct
        total_sequences += batch_size
        
        # Update progress bar
        token_acc = batch_tokens_correct / batch_tokens if batch_tokens > 0 else 0
        seq_acc = batch_sequences_correct / batch_size if batch_size > 0 else 0
        current_ppl = np.exp(-total_log_ppl / total_tokens) if total_tokens > 0 else 0
        
        master_pbar.set_postfix({
            'Batch': f'{batch_idx+1}/{len(dataloader)}',
            'Loss': f'{batch_loss.item():.2f}',
            'TokAcc': f'{token_acc:.2f}',
            'SeqAcc': f'{seq_acc:.2f}',
            'PPL': f'{current_ppl:.2f}',
            'TF': f'{teacher_forcing_ratio:.1f}'
        }, refresh=False)
        master_pbar.update(1)
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    token_accuracy = total_tokens_correct / total_tokens if total_tokens > 0 else 0
    sequence_accuracy = total_sequences_correct / total_sequences if total_sequences > 0 else 0
    perplexity = np.exp(-total_log_ppl / total_tokens) if total_tokens > 0 else 0
    
    return avg_loss, token_accuracy, sequence_accuracy, perplexity


def validate_epoch(model, dataloader, criterion, device):
    """Validation with full autoregressive prediction"""
    model.eval()
    total_loss, total_tokens_correct, total_tokens = 0, 0, 0
    total_sequences_correct, total_sequences = 0, 0
    total_log_ppl = 0
    
    with torch.no_grad():
        for inputs, targets, features, lengths in dataloader:
            inputs, targets, features = inputs.to(device), targets.to(device), features.to(device)
            lengths = lengths.to(device)
            
            batch_size, max_seq_len = inputs.shape
            
            # Forward pass through the entire sequence
            logits, _ = model(inputs, features, lengths.cpu())
            
            # Calculate metrics
            batch_tokens_correct = 0
            batch_tokens = 0
            batch_sequences_correct = 0
            batch_loss = 0
            
            for i in range(batch_size):
                seq_len_i = lengths[i].item()
                # Only consider non-padding positions
                seq_logits = logits[i, :seq_len_i]
                seq_targets = targets[i, :seq_len_i]
                
                # Loss
                seq_loss = criterion(seq_logits, seq_targets)
                batch_loss += seq_loss.item()
                
                # Token-level accuracy
                seq_preds = seq_logits.argmax(dim=-1)
                seq_correct = (seq_preds == seq_targets).sum().item()
                batch_tokens_correct += seq_correct
                batch_tokens += seq_len_i
                
                # Sequence-level accuracy
                if (seq_preds == seq_targets).all().item():
                    batch_sequences_correct += 1
                
                # Perplexity
                probs = torch.softmax(seq_logits, dim=-1)
                target_probs = probs.gather(-1, seq_targets.unsqueeze(-1)).squeeze(-1)
                total_log_ppl += torch.sum(torch.log(target_probs + 1e-8)).item()
            
            batch_loss = batch_loss / batch_size
            
            # Update totals
            total_loss += batch_loss
            total_tokens_correct += batch_tokens_correct
            total_tokens += batch_tokens
            total_sequences_correct += batch_sequences_correct
            total_sequences += batch_size
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    token_accuracy = total_tokens_correct / total_tokens if total_tokens > 0 else 0
    sequence_accuracy = total_sequences_correct / total_sequences if total_sequences > 0 else 0
    perplexity = np.exp(-total_log_ppl / total_tokens) if total_tokens > 0 else 0
    
    return avg_loss, token_accuracy, sequence_accuracy, perplexity


def calculate_baselines(train_dataset, val_dataset):
    """Calculate baselines for full sequence prediction"""
    # Get vocabulary info
    vocab = train_dataset.vocab
    vocab_size = len(vocab)
    
    # Count token frequencies in training data
    all_train_tokens = []
    for seq in train_dataset.sequences:
        all_train_tokens.extend(seq)
    token_counter = Counter(all_train_tokens)
    
    # Most frequent token
    most_frequent_token = token_counter.most_common(1)[0][0]
    most_frequent_count = token_counter[most_frequent_token]
    
    # Calculate probabilities for random baseline
    total_tokens = len(all_train_tokens)
    token_probs = {token: count / total_tokens for token, count in token_counter.items()}
    
    # Initialize results
    mode_token_correct = 0
    mode_sequence_correct = 0
    random_token_correct = 0
    random_sequence_correct = 0
    previous_token_correct = 0
    previous_sequence_correct = 0
    
    total_val_tokens = 0
    total_val_sequences = len(val_dataset.sequences)
    
    # Process validation sequences
    for seq in val_dataset.sequences:
        seq_len = len(seq)
        total_val_tokens += seq_len
        
        # Mode baseline: always predict most frequent token
        mode_predictions = [most_frequent_token] * seq_len
        mode_token_match = sum(1 for pred, true in zip(mode_predictions, seq) if pred == true)
        mode_token_correct += mode_token_match
        if mode_token_match == seq_len:
            mode_sequence_correct += 1
        
        # Random baseline: predict random tokens based on training distribution
        random_predictions = np.random.choice(
            list(token_probs.keys()), 
            size=seq_len,
            p=list(token_probs.values())
        )
        random_token_match = sum(1 for pred, true in zip(random_predictions, seq) if pred == true)
        random_token_correct += random_token_match
        if random_token_match == seq_len:
            random_sequence_correct += 1
        
        # Previous token baseline: predict the previous token
        previous_predictions = ['<START>'] + seq[:-1]  # First prediction is <START>, then previous tokens
        previous_token_match = sum(1 for pred, true in zip(previous_predictions, seq) if pred == true)
        previous_token_correct += previous_token_match
        if previous_token_match == seq_len:
            previous_sequence_correct += 1
    
    # Calculate accuracies
    mode_token_acc = mode_token_correct / total_val_tokens if total_val_tokens > 0 else 0
    mode_seq_acc = mode_sequence_correct / total_val_sequences if total_val_sequences > 0 else 0
    mode_ppl = np.exp(-np.log(mode_token_acc + 1e-8)) if mode_token_acc > 0 else float('inf')
    
    random_token_acc = random_token_correct / total_val_tokens if total_val_tokens > 0 else 0
    random_seq_acc = random_sequence_correct / total_val_sequences if total_val_sequences > 0 else 0
    random_ppl = vocab_size  # Perplexity of uniform distribution over vocabulary
    
    previous_token_acc = previous_token_correct / total_val_tokens if total_val_tokens > 0 else 0
    previous_seq_acc = previous_sequence_correct / total_val_sequences if total_val_sequences > 0 else 0
    previous_ppl = np.exp(-np.log(previous_token_acc + 1e-8)) if previous_token_acc > 0 else float('inf')
    
    print(f"\n{'='*80}")
    print(f"BASELINES (Full Sequence Prediction)")
    print(f"{'='*80}")
    print(f"Mode Baseline (always predict '{most_frequent_token}'):")
    print(f"  Token Accuracy: {mode_token_acc:.4f}, Sequence Accuracy: {mode_seq_acc:.4f}, Perplexity: {mode_ppl:.2f}")
    
    print(f"\nRandom Baseline (weighted by frequency):")
    print(f"  Token Accuracy: {random_token_acc:.4f}, Sequence Accuracy: {random_seq_acc:.4f}, Perplexity: {random_ppl:.2f}")
    
    print(f"\nPrevious Token Baseline:")
    print(f"  Token Accuracy: {previous_token_acc:.4f}, Sequence Accuracy: {previous_seq_acc:.4f}, Perplexity: {previous_ppl:.2f}")
    
    print(f"\nVocabulary size: {vocab_size}")
    print(f"Total validation sequences: {total_val_sequences}")
    print(f"Total validation tokens: {total_val_tokens}")
    
    return {
        'mode': {
            'token_acc': [mode_token_acc],
            'seq_acc': [mode_seq_acc],
            'ppl': [mode_ppl]
        },
        'random': {
            'token_acc': [random_token_acc],
            'seq_acc': [random_seq_acc],
            'ppl': [random_ppl]
        },
        'previous': {
            'token_acc': [previous_token_acc],
            'seq_acc': [previous_seq_acc],
            'ppl': [previous_ppl]
        }
    }


def train_model(model, train_loader, val_loader, optimizer, criterion, device, args):
    """Train with both standard and teacher forcing methods"""
    results = {
        'standard': {
            'train_loss': [], 'train_token_acc': [], 'train_seq_acc': [], 'train_ppl': [],
            'val_loss': [], 'val_token_acc': [], 'val_seq_acc': [], 'val_ppl': [],
        },
        'teacher_forcing': {
            'train_loss': [], 'train_token_acc': [], 'train_seq_acc': [], 'train_ppl': [],
            'val_loss': [], 'val_token_acc': [], 'val_seq_acc': [], 'val_ppl': [],
        }
    }
    
    # Train with both methods
    for method_name in ['standard', 'teacher_forcing']:
        print(f"\n{'='*80}")
        print(f"TRAINING WITH METHOD: {method_name.upper()}")
        print(f"{'='*80}")
        
        # Reset optimizer for each method
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=args.patience // 2, factor=0.5,
        )
        
        total_iterations = args.epochs * len(train_loader)
        master_pbar = tqdm(total=total_iterations, desc=f'TRAINING ({method_name.upper()})', 
                  position=0, leave=True, 
                  bar_format='{desc} {percentage:1.0f}%|{bar:5}| {n_fmt}/{total_fmt} {postfix}',
                  ncols=120)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(args.epochs):
            epoch_start = time.time()
            
            # Training with current method
            if method_name == 'teacher_forcing':
                train_loss, train_token_acc, train_seq_acc, train_ppl = train_epoch_teacher_forcing(
                    model, train_loader, optimizer, criterion, device, master_pbar, epoch, args.teacher_forcing_ratio)
            else:
                train_loss, train_token_acc, train_seq_acc, train_ppl = train_epoch_standard(
                    model, train_loader, optimizer, criterion, device, master_pbar, epoch)
            
            # Validation
            val_loss, val_token_acc, val_seq_acc, val_ppl = validate_epoch(model, val_loader, criterion, device)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start
            
            # Store results
            results[method_name]['train_loss'].append(train_loss)
            results[method_name]['train_token_acc'].append(train_token_acc)
            results[method_name]['train_seq_acc'].append(train_seq_acc)
            results[method_name]['train_ppl'].append(train_ppl)
            results[method_name]['val_loss'].append(val_loss)
            results[method_name]['val_token_acc'].append(val_token_acc)
            results[method_name]['val_seq_acc'].append(val_seq_acc)
            results[method_name]['val_ppl'].append(val_ppl)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
            
            master_pbar.set_postfix({
                'Epoch': f'{epoch+1}/{args.epochs}',
                'Trn_Tok': f'{train_token_acc:.2f}',
                'Trn_Seq': f'{train_seq_acc:.2f}',
                'Val_Tok': f'{val_token_acc:.2f}',
                'Val_Seq': f'{val_seq_acc:.2f}',
                'Val_PPL': f'{val_ppl:.2f}',
                'LR': f'{current_lr:.2e}',
                'Time': f'{epoch_time:.1f}s'
            }, refresh=False)
        
        master_pbar.close()
        
        print(f"\n{method_name.upper()} RESULTS:")
        print(f"Train - Token Acc: {train_token_acc:.4f}, Seq Acc: {train_seq_acc:.4f}, PPL: {train_ppl:.4f}")
        print(f"Val   - Token Acc: {val_token_acc:.4f}, Seq Acc: {val_seq_acc:.4f}, PPL: {val_ppl:.4f}")
        print(f"Best Val Loss: {best_val_loss:.4f}")
    
    return results


def run_experiment(args, device):
    # Create training dataset
    print("Creating training dataset...")
    train_dataset = AGEDataset(
        max_seq_length=args.max_seq_length,
        min_seq_length=args.min_seq_length,
        data_url=args.data_url,
        data_dir=args.data_dir,
        max_transactions=args.max_transactions,
        split_type='train',
        split_ratio=args.train_ratio,
        min_frequency=args.min_frequency
    )
    
    # Create validation dataset using training vocab and scaler
    print("Creating validation dataset...")
    val_dataset = AGEDataset(
        max_seq_length=args.max_seq_length,
        min_seq_length=args.min_seq_length,
        data_url=args.data_url,
        data_dir=args.data_dir,
        max_transactions=args.max_transactions,
        split_type='val',
        split_ratio=args.train_ratio,
        scaler=train_dataset.amount_scaler,
        vocab=train_dataset.vocab,
        min_frequency=args.min_frequency
    )
    
    # Statistics
    print(f"\nDATASET STATISTICS:")
    print(f"Training sequences: {len(train_dataset)}")
    print(f"Validation sequences: {len(val_dataset)}")
    print(f"Max sequence length: {args.max_seq_length}")
    print(f"Min sequence length: {args.min_seq_length}")
    print(f"Vocabulary size: {train_dataset.vocab_size}")
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("ERROR: One of the datasets is empty!")
        return {}
    
    # Calculate baselines
    baselines = calculate_baselines(train_dataset, val_dataset)
    
    # Create model
    model = NextTokenRNN(
        vocab_size=train_dataset.vocab_size,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        continuous_dim=2,
        rnn_type=args.rnn_type,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Use AdamW with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    
    print(f"\nMODEL CONFIGURATION:")
    print(f"Model: {args.rnn_type.upper()}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Teacher forcing ratio: {args.teacher_forcing_ratio}")
    print(f"Early stopping patience: {args.patience}")
    
    # Train model with both methods
    results = train_model(model, train_loader, val_loader, optimizer, criterion, device, args)
    
    # Prepare final report
    final_results = {
        'rnn_results': results,
        'baselines': baselines
    }
    
    return final_results


def print_final_report(results, baselines):
    """Print final comparison table"""
    print(f"\n{'='*100}")
    print("FINAL RESULTS COMPARISON")
    print(f"{'='*100}")
    
    headers = ["Method", "Token Acc", "Seq Acc", "Perplexity"]
    print(f"{headers[0]:<25} {headers[1]:<12} {headers[2]:<12} {headers[3]:<12}")
    print("-" * 65)
    
    # RNN Models
    for method_name in ['standard', 'teacher_forcing']:
        if method_name in results:
            rnn_results = results[method_name]
            token_acc = rnn_results['val_token_acc'][-1] if rnn_results['val_token_acc'] else 0
            seq_acc = rnn_results['val_seq_acc'][-1] if rnn_results['val_seq_acc'] else 0
            ppl = rnn_results['val_ppl'][-1] if rnn_results['val_ppl'] else 0
            print(f"{method_name.upper():<25} {token_acc:12.4f} {seq_acc:12.4f} {ppl:12.4f}")
    
    # Baselines
    for baseline_name in ['mode', 'random', 'previous']:
        if baseline_name in baselines:
            baseline = baselines[baseline_name]
            token_acc = baseline['token_acc'][0] if 'token_acc' in baseline else 0
            seq_acc = baseline['seq_acc'][0] if 'seq_acc' in baseline else 0
            ppl = baseline['ppl'][0] if 'ppl' in baseline else 0
            print(f"{baseline_name.capitalize() + ' Baseline':<25} {token_acc:12.4f} {seq_acc:12.4f} {ppl:12.4f}")


def main():
    args = parse_args()
    
    if not args.no_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    
    print("=" * 80)
    print("FULL AUTOREGRESSIVE NEXT TOKEN PREDICTION FOR AGE FINANCIAL TRANSACTIONS")
    print("=" * 80)
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("=" * 80)
        
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    print("Starting experiment...")
    results = run_experiment(args, device)
    
    if results:
        print_final_report(results['rnn_results'], results['baselines'])


if __name__ == "__main__":
    main()
