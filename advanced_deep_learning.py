import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                            roc_auc_score, roc_curve, f1_score, precision_recall_curve)
from pathlib import Path
import gzip
import os
import warnings
from datetime import datetime
import json
warnings.filterwarnings('ignore')

class GeneExpressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================================
# ADVANCED MODEL ARCHITECTURES
# ============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out += residual
        out = F.relu(out)
        return out


class DeepResNet(nn.Module):
    def __init__(self, input_dim, num_classes=3, hidden_dim=512, n_blocks=3, dropout_rate=0.4):
        super(DeepResNet, self).__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout_rate) for _ in range(n_blocks)
        ])
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.output_layer(x)
        return x


class AttentionMLP(nn.Module):
    def __init__(self, input_dim, num_classes=3, hidden_dim=512, dropout_rate=0.4):
        super(AttentionMLP, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        
        # Self-attention
        attention_weights = torch.softmax(self.attention(encoded), dim=0)
        attended = encoded * attention_weights
        
        output = self.classifier(attended)
        return output


class WideAndDeep(nn.Module):
    def __init__(self, input_dim, num_classes=3, dropout_rate=0.4):
        super(WideAndDeep, self).__init__()
        
        # Wide component (linear)
        self.wide = nn.Linear(input_dim, num_classes)
        
        # Deep component
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        wide_out = self.wide(x)
        deep_out = self.deep(x)
        return wide_out + deep_out


class AdvancedAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=128, dropout_rate=0.3):
        super(AdvancedAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, encoding_dim),
            nn.BatchNorm1d(encoding_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            
            nn.Linear(1024, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


# ============================================================================
# MAIN PIPELINE CLASS
# ============================================================================

class AdvancedDeepLearningPipeline:
    def __init__(self, expression_data, metadata, output_dir='./dl_advanced_results'):
        self.expr = expression_data
        self.metadata = metadata
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        
        print("="*80)
        print("🚀 ADVANCED DEEP LEARNING PIPELINE")
        print("="*80)
        print(f"\n📊 Data:")
        print(f"   Expression: {self.expr.shape[0]} genes × {self.expr.shape[1]} samples")
        print(f"   Metadata: {self.metadata.shape[0]} samples")
        
        self.models = {}
        self.results = {}
        self.training_history = {}
    
    def prepare_data(self, group_column='diabetes_label', exclude_classes=['T3cD'], 
                    n_genes=5000, test_size=0.2):
        print(f"\n🔧 DATA PREPARATION")
        print("-"*80)
        
        if 'title' in self.metadata.columns:
            import re
            def extract_sample_id(title):
                match = re.search(r'(DP\d+)', str(title))
                return match.group(1) if match else None
            
            self.metadata['sample_id'] = self.metadata['title'].apply(extract_sample_id)
            valid_mask = self.metadata['sample_id'].notna()
            self.metadata = self.metadata[valid_mask]
            self.metadata = self.metadata.set_index('sample_id', drop=False)
        
        common_samples = list(set(self.expr.columns) & set(self.metadata.index))
        self.expr = self.expr[common_samples]
        self.metadata = self.metadata.loc[common_samples]
        
        if exclude_classes:
            mask = ~self.metadata[group_column].isin(exclude_classes)
            self.metadata = self.metadata[mask]
            filtered_samples = self.metadata.index.tolist()
            self.expr = self.expr[filtered_samples]
        
        print(f"✅ Filtered {len(filtered_samples)} samples")
        print(self.metadata[group_column].value_counts())
        
        gene_vars = self.expr.var(axis=1).sort_values(ascending=False)
        top_genes = gene_vars.head(n_genes).index
        self.expr = self.expr.loc[top_genes]
        
        print(f"\n🧬 Selected top {len(top_genes)} variable genes")
        
        self.expr = self.expr.T
        self.expr = self.expr.fillna(self.expr.mean())
        self.expr = self.expr.replace([np.inf, -np.inf], 0)
        
        self.X = self.expr.values
        self.y = self.metadata[group_column].values
        
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y_encoded,
            test_size=test_size,
            random_state=42,
            stratify=self.y_encoded
        )
        
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test
        
        X_train_part, X_val, y_train_part, y_val = train_test_split(
            self.X_train_scaled, self.y_train,
            test_size=0.2,
            random_state=42,
            stratify=self.y_train
        )
        
        self.X_train_part = X_train_part
        self.X_val = X_val
        self.y_train_part = y_train_part
        self.y_val = y_val
        
        print(f"\n✅ Data splits:")
        print(f"   Train: {len(X_train_part)}")
        print(f"   Validation: {len(X_val)}")
        print(f"   Test: {len(self.X_test_scaled)}")
        print(f"   Features: {self.X.shape[1]}")
        print(f"   Classes: {self.label_encoder.classes_}")
    
    def train_model_advanced(self, model, model_name, 
                           epochs=200, batch_size=16, lr=0.001,
                           scheduler_type='cosine', use_label_smoothing=False):
        
        print(f"\n{'='*80}")
        print(f"🎯 TRAINING: {model_name}")
        print('='*80)
        
        train_dataset = GeneExpressionDataset(self.X_train_part, self.y_train_part)
        val_dataset = GeneExpressionDataset(self.X_val, self.y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Loss function
        if use_label_smoothing:
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Scheduler
        if scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        elif scheduler_type == 'onecycle':
            scheduler = OneCycleLR(optimizer, max_lr=lr*10, 
                                  steps_per_epoch=len(train_loader), 
                                  epochs=epochs)
        else:
            scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=15, factor=0.5)
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }
        
        best_val_acc = 0
        patience_counter = 0
        patience = 30
        
        print(f"Configuration:")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Initial LR: {lr}")
        print(f"   Scheduler: {scheduler_type}")
        print(f"   Label smoothing: {use_label_smoothing}")
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                if scheduler_type == 'onecycle':
                    scheduler.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            
            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total
            
            # Update scheduler
            current_lr = optimizer.param_groups[0]['lr']
            if scheduler_type == 'cosine':
                scheduler.step()
            elif scheduler_type == 'plateau':
                scheduler.step(val_acc)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(current_lr)
            
            # Print progress
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
                      f"LR: {current_lr:.6f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), 
                          f'{self.output_dir}/best_{model_name.replace(" ", "_")}.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        model.load_state_dict(torch.load(
            f'{self.output_dir}/best_{model_name.replace(" ", "_")}.pth'))
        
        print(f"\n✅ Training complete - Best Val Acc: {best_val_acc:.4f}")
        
        return model, history
    
    def evaluate_model_comprehensive(self, model, model_name):
        print(f"\n📊 EVALUATING: {model_name}")
        print("-"*80)
        
        model.eval()
        
        X_test_tensor = torch.FloatTensor(self.X_test_scaled).to(self.device)
        
        with torch.no_grad():
            outputs = model(X_test_tensor)
            probas = torch.softmax(outputs, dim=1).cpu().numpy()
            _, predicted = torch.max(outputs.data, 1)
            y_pred = predicted.cpu().numpy()
        
        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        f1_macro = f1_score(self.y_test, y_pred, average='macro')
        f1_weighted = f1_score(self.y_test, y_pred, average='weighted')
        
        try:
            auc = roc_auc_score(self.y_test, probas, multi_class='ovr', average='weighted')
        except:
            auc = None
        
        print(f"\n   Accuracy: {accuracy:.4f}")
        print(f"   F1-Score (Macro): {f1_macro:.4f}")
        print(f"   F1-Score (Weighted): {f1_weighted:.4f}")
        if auc:
            print(f"   ROC-AUC: {auc:.4f}")
        
        print(f"\n   Classification Report:")
        print(classification_report(self.y_test, y_pred, 
                                   target_names=self.label_encoder.classes_))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        self.plot_confusion_matrix(cm, model_name)
        
        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': probas,
            'confusion_matrix': cm
        }
    
    def plot_confusion_matrix(self, cm, model_name):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title(f'{model_name} - Confusion Matrix', fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/cm_{model_name.replace(" ", "_")}.png', dpi=300)
        plt.close()
    
    def plot_training_history(self, history, model_name):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Loss
        ax = axes[0]
        ax.plot(history['train_loss'], label='Train', linewidth=2)
        ax.plot(history['val_loss'], label='Validation', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{model_name} - Loss', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Accuracy
        ax = axes[1]
        ax.plot(history['train_acc'], label='Train', linewidth=2)
        ax.plot(history['val_acc'], label='Validation', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{model_name} - Accuracy', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning rate
        ax = axes[2]
        ax.plot(history['lr'], linewidth=2, color='green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title(f'{model_name} - Learning Rate', fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/history_{model_name.replace(" ", "_")}.png', dpi=300)
        plt.close()
    
    def model_ensemble_prediction(self):
        print(f"\n🎭 ENSEMBLE PREDICTION")
        print("-"*80)
        
        all_probas = []
        
        for name, result in self.results.items():
            if 'probabilities' in result:
                all_probas.append(result['probabilities'])
        
        if len(all_probas) == 0:
            print("No models with probabilities available")
            return None
        
        # Average probabilities
        ensemble_probas = np.mean(all_probas, axis=0)
        ensemble_pred = np.argmax(ensemble_probas, axis=1)
        
        # Evaluate
        accuracy = accuracy_score(self.y_test, ensemble_pred)
        f1_weighted = f1_score(self.y_test, ensemble_pred, average='weighted')
        
        print(f"   Ensemble Accuracy: {accuracy:.4f}")
        print(f"   Ensemble F1-Score: {f1_weighted:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, ensemble_pred)
        self.plot_confusion_matrix(cm, 'Ensemble')
        
        return {
            'model_name': 'Ensemble',
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'predictions': ensemble_pred,
            'probabilities': ensemble_probas
        }
    
    def plot_model_comparison(self):
        print(f"\n📊 Creating model comparison...")
        
        comparison_data = []
        for name, result in self.results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': result['accuracy'],
                'F1-Weighted': result['f1_weighted']
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Accuracy', ascending=False)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Accuracy
        ax = axes[0]
        x = range(len(df))
        ax.bar(x, df['Accuracy'], alpha=0.7, color='steelblue')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Model'], rotation=45, ha='right')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Accuracy Comparison', fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(df['Accuracy']):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
        
        # F1-Score
        ax = axes[1]
        ax.bar(x, df['F1-Weighted'], alpha=0.7, color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Model'], rotation=45, ha='right')
        ax.set_ylabel('F1-Score (Weighted)')
        ax.set_title('Model F1-Score Comparison', fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, v in enumerate(df['F1-Weighted']):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/model_comparison.png', dpi=300)
        plt.close()
        
        df.to_csv(f'{self.output_dir}/model_comparison.csv', index=False)
        
        print(f"   💾 Saved comparison")
    
    def generate_comprehensive_report(self):
        print(f"\n📋 GENERATING COMPREHENSIVE REPORT")
        print("-"*80)
        
        report = []
        
        report.append("="*100)
        report.append("ADVANCED DEEP LEARNING CLASSIFICATION REPORT")
        report.append("="*100)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Device: {self.device}")
        report.append("")
        
        report.append("="*100)
        report.append("1. DATASET SUMMARY")
        report.append("="*100)
        report.append(f"Total Samples: {len(self.X)}")
        report.append(f"Training Samples: {len(self.X_train_part)}")
        report.append(f"Validation Samples: {len(self.X_val)}")
        report.append(f"Test Samples: {len(self.X_test_scaled)}")
        report.append(f"Features: {self.X.shape[1]}")
        report.append(f"Classes: {', '.join(self.label_encoder.classes_)}")
        report.append("")
        
        report.append("="*100)
        report.append("2. MODEL PERFORMANCE")
        report.append("="*100)
        
        results_df = pd.DataFrame([
            {
                'Model': r['model_name'],
                'Accuracy': r['accuracy'],
                'F1-Macro': r.get('f1_macro', 0),
                'F1-Weighted': r['f1_weighted'],
                'ROC-AUC': r.get('auc', 0) if r.get('auc') else 0
            }
            for r in self.results.values()
        ])
        results_df = results_df.sort_values('Accuracy', ascending=False)
        
        report.append(results_df.to_string(index=False))
        report.append("")
        
        report.append("="*100)
        report.append("3. BEST MODEL")
        report.append("="*100)
        best = results_df.iloc[0]
        report.append(f"Model: {best['Model']}")
        report.append(f"Accuracy: {best['Accuracy']:.4f}")
        report.append(f"F1-Score (Weighted): {best['F1-Weighted']:.4f}")
        if best['ROC-AUC'] > 0:
            report.append(f"ROC-AUC: {best['ROC-AUC']:.4f}")
        report.append("")
        
        report.append("="*100)
        report.append("4. TRAINING INSIGHTS")
        report.append("="*100)
        for name, history in self.training_history.items():
            best_epoch = np.argmax(history['val_acc'])
            report.append(f"\n{name}:")
            report.append(f"  Best Epoch: {best_epoch + 1}")
            report.append(f"  Best Val Acc: {history['val_acc'][best_epoch]:.4f}")
            report.append(f"  Final LR: {history['lr'][-1]:.6f}")
        report.append("")
        
        report.append("="*100)
        report.append("5. RECOMMENDATIONS")
        report.append("="*100)
        
        if best['Accuracy'] > 0.85:
            report.append("- Excellent performance achieved (>85%)")
        elif best['Accuracy'] > 0.75:
            report.append("- Good performance (75-85%)")
        else:
            report.append("- Consider more data or architecture tuning (<75%)")
        
        report.append("- Ensemble prediction combines multiple models for robustness")
        report.append("- Monitor overfitting through training curves")
        report.append("- Consider data augmentation for improvement")
        
        report.append("")
        report.append("="*100)
        
        report_text = "\n".join(report)
        print(report_text)
        
        with open(f'{self.output_dir}/comprehensive_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n💾 Saved comprehensive report")
    
    def run_full_pipeline(self):
        print("\n" + "="*80)
        print("🚀 RUNNING FULL ADVANCED DL PIPELINE")
        print("="*80)
        
        # Prepare data
        self.prepare_data(n_genes=5000)
        
        input_dim = self.X_train_part.shape[1]
        num_classes = len(self.label_encoder.classes_)
        
        # Model 1: Deep ResNet
        print(f"\n{'='*80}")
        print("MODEL 1: Deep ResNet")
        print('='*80)
        
        resnet = DeepResNet(input_dim, num_classes, hidden_dim=512, 
                           n_blocks=3, dropout_rate=0.4).to(self.device)
        resnet, resnet_history = self.train_model_advanced(
            resnet, 'DeepResNet',
            epochs=200, batch_size=16, lr=0.001,
            scheduler_type='cosine', use_label_smoothing=True
        )
        self.models['DeepResNet'] = resnet
        self.training_history['DeepResNet'] = resnet_history
        self.plot_training_history(resnet_history, 'DeepResNet')
        
        resnet_result = self.evaluate_model_comprehensive(resnet, 'DeepResNet')
        self.results['DeepResNet'] = resnet_result
        
        # Model 2: Attention MLP
        print(f"\n{'='*80}")
        print("MODEL 2: Attention MLP")
        print('='*80)
        
        attn_mlp = AttentionMLP(input_dim, num_classes, 
                               hidden_dim=512, dropout_rate=0.4).to(self.device)
        attn_mlp, attn_history = self.train_model_advanced(
            attn_mlp, 'AttentionMLP',
            epochs=200, batch_size=16, lr=0.001,
            scheduler_type='cosine', use_label_smoothing=True
        )
        self.models['AttentionMLP'] = attn_mlp
        self.training_history['AttentionMLP'] = attn_history
        self.plot_training_history(attn_history, 'AttentionMLP')
        
        attn_result = self.evaluate_model_comprehensive(attn_mlp, 'AttentionMLP')
        self.results['AttentionMLP'] = attn_result
        
        # Model 3: Wide & Deep
        print(f"\n{'='*80}")
        print("MODEL 3: Wide & Deep")
        print('='*80)
        
        wide_deep = WideAndDeep(input_dim, num_classes, dropout_rate=0.4).to(self.device)
        wide_deep, wd_history = self.train_model_advanced(
            wide_deep, 'WideAndDeep',
            epochs=200, batch_size=16, lr=0.001,
            scheduler_type='cosine', use_label_smoothing=False
        )
        self.models['WideAndDeep'] = wide_deep
        self.training_history['WideAndDeep'] = wd_history
        self.plot_training_history(wd_history, 'WideAndDeep')
        
        wd_result = self.evaluate_model_comprehensive(wide_deep, 'WideAndDeep')
        self.results['WideAndDeep'] = wd_result
        
        # Ensemble
        print(f"\n{'='*80}")
        print("ENSEMBLE MODEL")
        print('='*80)
        
        ensemble_result = self.model_ensemble_prediction()
        if ensemble_result:
            self.results['Ensemble'] = ensemble_result
        
        # Comparison
        self.plot_model_comparison()
        
        # Report
        self.generate_comprehensive_report()
        
        print("\n" + "="*80)
        print("✅ ADVANCED DL PIPELINE COMPLETE!")
        print("="*80)
        print(f"\n📁 Results saved in: {self.output_dir}/")
        print("\nGenerated files:")
        print("   - Model weights (*.pth)")
        print("   - Training histories (history_*.png)")
        print("   - Confusion matrices (cm_*.png)")
        print("   - Model comparison (model_comparison.png/csv)")
        print("   - Comprehensive report (comprehensive_report.txt)")
        print("="*80)


def load_gse164416_data():
    print("="*80)
    print("📥 LOADING GSE164416 DATA")
    print("="*80)
    
    print("\n🧬 Loading expression data...")
    expr_path_gz = './suppl_data/GSE164416/GSE164416_DP_htseq_counts.txt.gz'
    
    if not os.path.exists(expr_path_gz):
        raise FileNotFoundError(f"Expression file not found: {expr_path_gz}")
    
    with gzip.open(expr_path_gz, 'rt') as f:
        expression_data = pd.read_csv(f, sep='\t', index_col=0)
    
    print(f"✅ Loaded: {expression_data.shape[0]} genes × {expression_data.shape[1]} samples")
    
    print("\n📋 Loading metadata...")
    metadata_path = './outputdata/GSE164416_metadata.csv'
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    metadata = pd.read_csv(metadata_path, index_col=0)
    print(f"✅ Loaded: {metadata.shape[0]} samples")
    
    return expression_data, metadata


if __name__ == '__main__':
    
    expression_data, metadata = load_gse164416_data()
    
    pipeline = AdvancedDeepLearningPipeline(expression_data, metadata)
    
    pipeline.run_full_pipeline()
    
    print("\n" + "="*80)
    print("🎉 ALL DONE!")
    print("="*80)