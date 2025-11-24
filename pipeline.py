"""
ClinVar Variant Sensitivity Assessment Pipeline
Nucleotide Transformer v2 + Contrastive Learning
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
import pysam
from sklearn.model_selection import train_test_split

# ============================================================================
# 1. DATA PREPROCESSING
# ============================================================================

class ClinVarPreprocessor:
    """ClinVar VCF + Reference Genome Preprocessor"""
    
    def __init__(self, vcf_path, fasta_path, window_size=512):
        self.vcf_path = vcf_path
        self.fasta_path = fasta_path
        self.window_size = window_size
        self.half_window = window_size // 2
        
    def parse_clnsig(self, clnsig_str):
        """Parse CLNSIG field and return label"""
        if not clnsig_str or clnsig_str == '.':
            return None
        
        clnsig_lower = clnsig_str.lower()
        
        # Pathogenic (Label 1)
        if 'pathogenic' in clnsig_lower and 'benign' not in clnsig_lower:
            if 'likely_pathogenic' in clnsig_lower or 'likely pathogenic' in clnsig_lower:
                return 1
            if clnsig_lower == 'pathogenic':
                return 1
                
        # Benign (Label 0)
        if 'benign' in clnsig_lower and 'pathogenic' not in clnsig_lower:
            if 'likely_benign' in clnsig_lower or 'likely benign' in clnsig_lower:
                return 0
            if clnsig_lower == 'benign':
                return 0
        
        # Conflicting or Uncertain
        return None
    
    def extract_sequence(self, fasta, chrom, pos, ref, alt):
        """Extract reference and variant sequences"""
        try:
            # Adjust chromosome format
            if not chrom.startswith('chr'):
                chrom = f'chr{chrom}'
            
            # Extract window centered at variant position
            start = max(0, pos - self.half_window)
            end = pos + self.half_window
            
            ref_seq = fasta.fetch(chrom, start, end).upper()
            
            # Check if sequence length is correct
            if len(ref_seq) != self.window_size:
                return None, None
            
            # Create variant sequence (replace center nucleotide)
            var_seq = list(ref_seq)
            center_idx = self.half_window - 1  # 0-indexed
            
            # Verify reference allele matches
            if var_seq[center_idx] != ref:
                return None, None
            
            # Apply variant
            var_seq[center_idx] = alt
            var_seq = ''.join(var_seq)
            
            return ref_seq, var_seq
            
        except Exception as e:
            return None, None
    
    def preprocess(self, output_csv='clinvar_processed.csv', max_samples=None):
        """Main preprocessing pipeline"""
        print(f"🧬 Loading Reference Genome: {self.fasta_path}")
        fasta = pysam.FastaFile(self.fasta_path)
        
        print(f"📊 Parsing VCF: {self.vcf_path}")
        vcf = pysam.VariantFile(self.vcf_path)
        
        data = []
        skipped = {'label': 0, 'sequence': 0, 'other': 0}
        
        for idx, record in enumerate(tqdm(vcf, desc="Processing variants")):
            if max_samples and len(data) >= max_samples:
                break
                
            try:
                # Get CLNSIG
                clnsig = record.info.get('CLNSIG', [None])[0]
                label = self.parse_clnsig(clnsig)
                
                if label is None:
                    skipped['label'] += 1
                    continue
                
                # Only process SNVs (single nucleotide variants)
                ref = record.ref
                alt = record.alts[0] if record.alts else None
                
                if not alt or len(ref) != 1 or len(alt) != 1:
                    skipped['other'] += 1
                    continue
                
                # Extract sequences
                chrom = record.chrom
                pos = record.pos
                
                ref_seq, var_seq = self.extract_sequence(fasta, chrom, pos, ref, alt)
                
                if ref_seq is None or var_seq is None:
                    skipped['sequence'] += 1
                    continue
                
                data.append({
                    'id': f"{chrom}_{pos}_{ref}_{alt}",
                    'chrom': chrom,
                    'pos': pos,
                    'ref': ref,
                    'alt': alt,
                    'ref_seq': ref_seq,
                    'var_seq': var_seq,
                    'label': label
                })
                
            except Exception as e:
                skipped['other'] += 1
                continue
        
        fasta.close()
        vcf.close()
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        print(f"\n✅ Processing Complete!")
        print(f"   Total variants processed: {len(df)}")
        print(f"   Pathogenic (Label=1): {(df['label']==1).sum()}")
        print(f"   Benign (Label=0): {(df['label']==0).sum()}")
        print(f"   Skipped - No valid label: {skipped['label']}")
        print(f"   Skipped - Sequence extraction failed: {skipped['sequence']}")
        print(f"   Skipped - Other reasons: {skipped['other']}")
        
        # Save
        df.to_csv(output_csv, index=False)
        print(f"💾 Saved to: {output_csv}")
        
        return df


# ============================================================================
# 2. DATASET
# ============================================================================

class VariantDataset(Dataset):
    """Dataset for Reference-Variant sequence pairs"""
    
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        ref_seq = row['ref_seq']
        var_seq = row['var_seq']
        label = row['label']
        
        # Tokenize sequences
        ref_encoding = self.tokenizer(
            ref_seq,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        var_encoding = self.tokenizer(
            var_seq,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'ref_input_ids': ref_encoding['input_ids'].squeeze(0),
            'ref_attention_mask': ref_encoding['attention_mask'].squeeze(0),
            'var_input_ids': var_encoding['input_ids'].squeeze(0),
            'var_attention_mask': var_encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


class TestDataset(Dataset):
    """Dataset for test sequences (inference only)"""
    
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        seq = row['seq']
        seq_id = row['ID']
        
        encoding = self.tokenizer(
            seq,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'id': seq_id
        }


# ============================================================================
# 3. MODEL ARCHITECTURE
# ============================================================================

class NucleotideTransformerEmbedder(nn.Module):
    """Nucleotide Transformer v2 with Mean Pooling"""
    
    def __init__(self, model_name="InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", 
                 embedding_dim=2048):
        super().__init__()
        
        print(f"🔧 Loading Nucleotide Transformer: {model_name}")
        self.transformer = AutoModelForMaskedLM.from_pretrained(model_name)
        
        # Get hidden size from transformer config
        self.hidden_size = self.transformer.config.hidden_size
        self.embedding_dim = embedding_dim
        
        # Projection layer (hidden_size -> 2048)
        self.projector = nn.Linear(self.hidden_size, embedding_dim)
        
        print(f"   Hidden size: {self.hidden_size}")
        print(f"   Output embedding dim: {embedding_dim}")
    
    def mean_pooling(self, hidden_states, attention_mask):
        """Mean pooling over sequence length"""
        # hidden_states: (batch, seq_len, hidden_dim)
        # attention_mask: (batch, seq_len)
        
        # Expand attention mask
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        
        # Sum of hidden states
        sum_hidden = torch.sum(hidden_states * attention_mask_expanded, dim=1)
        
        # Sum of attention mask
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
        
        # Mean pooling
        mean_hidden = sum_hidden / sum_mask
        
        return mean_hidden
    
    def forward(self, input_ids, attention_mask):
        """Forward pass with mean pooling"""
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Use last hidden state
        hidden_states = outputs.hidden_states[-1]
        
        # Mean pooling
        pooled = self.mean_pooling(hidden_states, attention_mask)
        
        # Project to embedding dimension
        embeddings = self.projector(pooled)
        
        # L2 normalization for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


# ============================================================================
# 4. CONTRASTIVE LOSS FUNCTION
# ============================================================================

class ContrastiveCDDLoss(nn.Module):
    """
    Contrastive Loss for Cosine Distance Difference (CDD) Optimization
    
    - Benign (label=0): Minimize cosine distance (maximize similarity)
    - Pathogenic (label=1): Maximize cosine distance (minimize similarity)
    """
    
    def __init__(self, margin=1.0, benign_weight=1.0, pathogenic_weight=1.0):
        super().__init__()
        self.margin = margin
        self.benign_weight = benign_weight
        self.pathogenic_weight = pathogenic_weight
    
    def cosine_distance(self, emb1, emb2):
        """Compute cosine distance (1 - cosine_similarity)"""
        cos_sim = F.cosine_similarity(emb1, emb2, dim=1)
        cos_dist = 1 - cos_sim
        return cos_dist
    
    def forward(self, ref_emb, var_emb, labels):
        """
        Args:
            ref_emb: Reference embeddings (batch, embed_dim)
            var_emb: Variant embeddings (batch, embed_dim)
            labels: 0 for Benign, 1 for Pathogenic
        """
        cos_dist = self.cosine_distance(ref_emb, var_emb)
        
        # Benign: minimize distance (want cos_dist → 0)
        benign_mask = (labels == 0).float()
        benign_loss = cos_dist * benign_mask
        
        # Pathogenic: maximize distance (want cos_dist → margin)
        pathogenic_mask = (labels == 1).float()
        pathogenic_loss = torch.clamp(self.margin - cos_dist, min=0) * pathogenic_mask
        
        # Weighted sum
        total_loss = (
            self.benign_weight * benign_loss.sum() +
            self.pathogenic_weight * pathogenic_loss.sum()
        ) / (benign_mask.sum() + pathogenic_mask.sum() + 1e-8)
        
        return total_loss, {
            'benign_loss': benign_loss.sum() / (benign_mask.sum() + 1e-8),
            'pathogenic_loss': pathogenic_loss.sum() / (pathogenic_mask.sum() + 1e-8),
            'avg_cos_dist': cos_dist.mean()
        }


# ============================================================================
# 5. TRAINING
# ============================================================================

class Trainer:
    """Training Pipeline"""
    
    def __init__(self, model, train_loader, val_loader, device, 
                 lr=2e-5, epochs=5, margin=1.0):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.criterion = ContrastiveCDDLoss(margin=margin)
        
        self.best_val_loss = float('inf')
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
        
        for batch in pbar:
            ref_input_ids = batch['ref_input_ids'].to(self.device)
            ref_attention_mask = batch['ref_attention_mask'].to(self.device)
            var_input_ids = batch['var_input_ids'].to(self.device)
            var_attention_mask = batch['var_attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            ref_emb = self.model(ref_input_ids, ref_attention_mask)
            var_emb = self.model(var_input_ids, var_attention_mask)
            
            # Compute loss
            loss, metrics = self.criterion(ref_emb, var_emb, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'cos_dist': f"{metrics['avg_cos_dist'].item():.4f}"
            })
        
        return total_loss / len(self.train_loader)
    
    def validate(self, epoch):
        """Validate"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Val]")
            
            for batch in pbar:
                ref_input_ids = batch['ref_input_ids'].to(self.device)
                ref_attention_mask = batch['ref_attention_mask'].to(self.device)
                var_input_ids = batch['var_input_ids'].to(self.device)
                var_attention_mask = batch['var_attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                ref_emb = self.model(ref_input_ids, ref_attention_mask)
                var_emb = self.model(var_input_ids, var_attention_mask)
                
                # Compute loss
                loss, metrics = self.criterion(ref_emb, var_emb, labels)
                total_loss += loss.item()
                
                pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})
        
        avg_val_loss = total_loss / len(self.val_loader)
        return avg_val_loss
    
    def train(self, save_path='best_model.pt'):
        """Full training loop"""
        print(f"\n🚀 Starting Training...")
        print(f"   Device: {self.device}")
        print(f"   Epochs: {self.epochs}")
        print(f"   Train batches: {len(self.train_loader)}")
        print(f"   Val batches: {len(self.val_loader)}")
        
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            
            print(f"\n📊 Epoch {epoch+1}/{self.epochs}")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"   ✅ Best model saved! (Val Loss: {val_loss:.4f})")
        
        print(f"\n🎉 Training Complete!")
        print(f"   Best Val Loss: {self.best_val_loss:.4f}")


# ============================================================================
# 6. INFERENCE
# ============================================================================

def inference(model, test_loader, device, output_csv='submission.csv'):
    """Generate embeddings for test set"""
    print(f"\n🔮 Starting Inference...")
    
    model.eval()
    all_embeddings = []
    all_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating embeddings"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ids = batch['id']
            
            # Generate embeddings
            embeddings = model(input_ids, attention_mask)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_ids.extend(ids)
    
    # Concatenate all embeddings
    embeddings_array = np.vstack(all_embeddings)
    
    # Create submission dataframe
    submission = pd.DataFrame(embeddings_array)
    submission.columns = [f'emb_{i:04d}' for i in range(embeddings_array.shape[1])]
    submission.insert(0, 'ID', all_ids)
    
    # Save
    submission.to_csv(output_csv, index=False)
    print(f"✅ Submission saved to: {output_csv}")
    print(f"   Shape: {submission.shape}")
    
    return submission


# ============================================================================
# 7. MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline"""
    
    # Configuration
    CONFIG = {
        'vcf_path': 'clinvar.vcf.gz',
        'fasta_path': 'hg38.fa',
        'test_csv': 'test.csv',
        'processed_csv': 'clinvar_processed.csv',
        'model_name': 'InstaDeepAI/nucleotide-transformer-v2-500m-multi-species',
        'window_size': 512,
        'embedding_dim': 2048,
        'batch_size': 8,
        'lr': 2e-5,
        'epochs': 5,
        'margin': 1.0,
        'max_samples': None,  # Set to limit samples for testing
        'seed': 42
    }
        
    # Set seed
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Using device: {device}")
    
    # ========================================================================
    # STEP 1: Preprocess ClinVar Data
    # ========================================================================
    if not os.path.exists(CONFIG['processed_csv']):
        print("\n" + "="*80)
        print("STEP 1: DATA PREPROCESSING")
        print("="*80)
        
        preprocessor = ClinVarPreprocessor(
            vcf_path=CONFIG['vcf_path'],
            fasta_path=CONFIG['fasta_path'],
            window_size=CONFIG['window_size']
        )
        
        df = preprocessor.preprocess(
            output_csv=CONFIG['processed_csv'],
            max_samples=CONFIG['max_samples']
        )
    else:
        print(f"\n✅ Loading existing processed data: {CONFIG['processed_csv']}")
        df = pd.read_csv(CONFIG['processed_csv'])
        print(f"   Total samples: {len(df)}")
    
    # ========================================================================
    # STEP 2: Prepare Datasets
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: PREPARING DATASETS")
    print("="*80)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    print(f"✅ Tokenizer loaded")
    
    # Train/Val split
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=CONFIG['seed'], stratify=df['label']
    )
    print(f"   Train set: {len(train_df)} samples")
    print(f"   Val set: {len(val_df)} samples")
    
    # Create datasets
    train_dataset = VariantDataset(train_df, tokenizer, max_length=CONFIG['window_size'])
    val_dataset = VariantDataset(val_df, tokenizer, max_length=CONFIG['window_size'])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False,
        num_workers=2
    )
    
    # ========================================================================
    # STEP 3: Initialize Model
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: MODEL INITIALIZATION")
    print("="*80)
    
    model = NucleotideTransformerEmbedder(
        model_name=CONFIG['model_name'],
        embedding_dim=CONFIG['embedding_dim']
    )
    
    # ========================================================================
    # STEP 4: Training
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: TRAINING")
    print("="*80)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=CONFIG['lr'],
        epochs=CONFIG['epochs'],
        margin=CONFIG['margin']
    )
    
    trainer.train(save_path='best_model.pt')
    
    # ========================================================================
    # STEP 5: Inference on Test Set
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: INFERENCE")
    print("="*80)
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    print("✅ Best model loaded")
    
    # Load test data
    test_df = pd.read_csv(CONFIG['test_csv'])
    print(f"✅ Test data loaded: {len(test_df)} samples")
    
    # Create test dataset
    test_dataset = TestDataset(test_df, tokenizer, max_length=CONFIG['window_size'])
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    # Generate embeddings
    submission = inference(
        model=model,
        test_loader=test_loader,
        device=device,
        output_csv='submission.csv'
    )
    
    print("\n" + "="*80)
    print("🎊 PIPELINE COMPLETE!")
    print("="*80)
    print("Generated files:")
    print(f"  - {CONFIG['processed_csv']}")
    print(f"  - best_model.pt")
    print(f"  - submission.csv")


if __name__ == '__main__':
    main()
