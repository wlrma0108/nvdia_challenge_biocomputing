import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from pathlib import Path
import gzip
import os
import warnings
warnings.filterwarnings('ignore')

class GeneExpressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], num_classes=3, dropout_rate=0.5):
        super(SimpleMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class GeneConv1D(nn.Module):
    def __init__(self, input_dim, num_classes=3, dropout_rate=0.5):
        super(GeneConv1D, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.squeeze(-1)
        x = self.classifier(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=128):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, encoding_dim)
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


class AutoencoderClassifier(nn.Module):
    def __init__(self, autoencoder, encoding_dim=128, num_classes=3, dropout_rate=0.5):
        super(AutoencoderClassifier, self).__init__()
        
        self.encoder = autoencoder.encoder
        
        for param in self.encoder.parameters():
            param.requires_grad = True
        
        self.classifier = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        output = self.classifier(encoded)
        return output


class DeepLearningPipeline:
    def __init__(self, expression_data, metadata, output_dir='./dl_results'):
        self.expr = expression_data
        self.metadata = metadata
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        print("="*80)
        print("🤖 DEEP LEARNING PIPELINE")
        print("="*80)
        print(f"\n📊 Data:")
        print(f"   Expression: {self.expr.shape[0]} genes × {self.expr.shape[1]} samples")
        print(f"   Metadata: {self.metadata.shape[0]} samples")
    
    def prepare_data(self, group_column='diabetes_label', exclude_classes=['T3cD'], n_genes=5000):
        print(f"\n🔧 PREPARING DATA")
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
        
        mask = ~self.metadata[group_column].isin(exclude_classes)
        self.metadata = self.metadata[mask]
        filtered_samples = self.metadata.index.tolist()
        self.expr = self.expr[filtered_samples]
        
        print(f"✅ Matched {len(filtered_samples)} samples")
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
        
        print(f"\n✅ Final shape: {self.X.shape}")
        print(f"   Classes: {self.label_encoder.classes_}")
    
    def train_autoencoder(self, encoding_dim=128, epochs=100, batch_size=16, lr=0.001):
        print(f"\n🔄 TRAINING AUTOENCODER")
        print("-"*80)
        
        X_train, X_val = train_test_split(self.X, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.FloatTensor(X_train_scaled)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_scaled),
            torch.FloatTensor(X_val_scaled)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        autoencoder = Autoencoder(self.X.shape[1], encoding_dim).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(autoencoder.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
        
        train_losses = []
        val_losses = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        for epoch in range(epochs):
            autoencoder.train()
            train_loss = 0
            
            for batch_X, batch_X_target in train_loader:
                batch_X = batch_X.to(self.device)
                batch_X_target = batch_X_target.to(self.device)
                
                optimizer.zero_grad()
                
                decoded, _ = autoencoder(batch_X)
                loss = criterion(decoded, batch_X_target)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            autoencoder.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_X, batch_X_target in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_X_target = batch_X_target.to(self.device)
                    
                    decoded, _ = autoencoder(batch_X)
                    loss = criterion(decoded, batch_X_target)
                    
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(autoencoder.state_dict(), f'{self.output_dir}/best_autoencoder.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"   Early stopping at epoch {epoch+1}")
                break
        
        autoencoder.load_state_dict(torch.load(f'{self.output_dir}/best_autoencoder.pth'))
        
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Autoencoder Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{self.output_dir}/autoencoder_training.png', dpi=300)
        plt.close()
        
        print(f"\n✅ Autoencoder trained - Best Val Loss: {best_val_loss:.4f}")
        
        self.autoencoder = autoencoder
        self.scaler = scaler
        
        return autoencoder, scaler
    
    def train_model(self, model, model_name, X_train, y_train, X_val, y_val, 
                   epochs=200, batch_size=16, lr=0.001):
        
        train_dataset = GeneExpressionDataset(X_train, y_train)
        val_dataset = GeneExpressionDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=15, factor=0.5)
        
        train_accs = []
        val_accs = []
        train_losses = []
        val_losses = []
        
        best_val_acc = 0
        patience_counter = 0
        patience = 30
        
        for epoch in range(epochs):
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
                optimizer.step()
                
                train_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = train_correct / train_total
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
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
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            scheduler.step(val_acc)
            
            if (epoch + 1) % 20 == 0:
                print(f"   Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), f'{self.output_dir}/best_{model_name}.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"   Early stopping at epoch {epoch+1}")
                break
        
        model.load_state_dict(torch.load(f'{self.output_dir}/best_{model_name}.pth'))
        
        return model, train_accs, val_accs, train_losses, val_losses
    
    def evaluate_model(self, model, model_name, X_test, y_test):
        print(f"\n📊 EVALUATING {model_name}")
        print("-"*80)
        
        model.eval()
        
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        with torch.no_grad():
            outputs = model(X_test_tensor)
            probas = torch.softmax(outputs, dim=1).cpu().numpy()
            _, predicted = torch.max(outputs.data, 1)
            y_pred = predicted.cpu().numpy()
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_
        )
        plt.title(f'{model_name} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/{model_name}_confusion_matrix.png', dpi=300)
        plt.close()
        
        accuracy = accuracy_score(y_test, y_pred)
        
        try:
            auc = roc_auc_score(y_test, probas, multi_class='ovr', average='weighted')
            print(f"\nAccuracy: {accuracy:.3f}")
            print(f"ROC-AUC: {auc:.3f}")
        except:
            print(f"\nAccuracy: {accuracy:.3f}")
            auc = None
        
        return {
            'model': model_name,
            'accuracy': accuracy,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': probas
        }
    
    def plot_training_curves(self, results_dict):
        print(f"\n📈 Creating training curves...")
        
        n_models = len(results_dict)
        fig, axes = plt.subplots(n_models, 2, figsize=(14, n_models*4))
        
        if n_models == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (model_name, data) in enumerate(results_dict.items()):
            axes[idx, 0].plot(data['train_accs'], label='Train')
            axes[idx, 0].plot(data['val_accs'], label='Validation')
            axes[idx, 0].set_xlabel('Epoch')
            axes[idx, 0].set_ylabel('Accuracy')
            axes[idx, 0].set_title(f'{model_name} - Accuracy')
            axes[idx, 0].legend()
            axes[idx, 0].grid(True, alpha=0.3)
            
            axes[idx, 1].plot(data['train_losses'], label='Train')
            axes[idx, 1].plot(data['val_losses'], label='Validation')
            axes[idx, 1].set_xlabel('Epoch')
            axes[idx, 1].set_ylabel('Loss')
            axes[idx, 1].set_title(f'{model_name} - Loss')
            axes[idx, 1].legend()
            axes[idx, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/training_curves.png', dpi=300)
        plt.close()
        
        print(f"   💾 Saved training curves")
    
    def run_kfold_evaluation(self, model_class, model_name, model_params, n_splits=5):
        print(f"\n🔄 K-FOLD CROSS-VALIDATION: {model_name}")
        print("="*80)
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        fold_results = []
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, self.y_encoded), 1):
            print(f"\n📊 Fold {fold}/{n_splits}")
            print("-"*40)
            
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = self.y_encoded[train_idx], self.y_encoded[val_idx]
            
            model = model_class(**model_params).to(self.device)
            
            model, train_accs, val_accs, train_losses, val_losses = self.train_model(
                model, f'{model_name}_fold{fold}',
                X_train, y_train, X_val, y_val,
                epochs=150, batch_size=16, lr=0.001
            )
            
            model.eval()
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            
            with torch.no_grad():
                outputs = model(X_val_tensor)
                _, predicted = torch.max(outputs.data, 1)
                y_pred = predicted.cpu().numpy()
            
            accuracy = accuracy_score(y_val, y_pred)
            fold_results.append(accuracy)
            
            print(f"   Fold {fold} Accuracy: {accuracy:.3f}")
        
        mean_acc = np.mean(fold_results)
        std_acc = np.std(fold_results)
        
        print(f"\n✅ {model_name} K-Fold Results:")
        print(f"   Mean Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
        print(f"   Individual Folds: {[f'{acc:.3f}' for acc in fold_results]}")
        
        return fold_results, mean_acc, std_acc
    
    def run_full_pipeline(self):
        print("\n" + "="*80)
        print("🚀 RUNNING FULL DEEP LEARNING PIPELINE")
        print("="*80)
        
        self.prepare_data(n_genes=5000)
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y_encoded,
            test_size=0.2,
            random_state=42,
            stratify=self.y_encoded
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_part, X_val, y_train_part, y_val = train_test_split(
            X_train_scaled, y_train,
            test_size=0.2,
            random_state=42,
            stratify=y_train
        )
        
        print(f"\n📊 Data splits:")
        print(f"   Train: {len(X_train_part)}")
        print(f"   Val: {len(X_val)}")
        print(f"   Test: {len(X_test_scaled)}")
        
        autoencoder, ae_scaler = self.train_autoencoder(encoding_dim=128, epochs=100)
        
        print(f"\n{'='*80}")
        print("🤖 MODEL 1: SIMPLE MLP")
        print('='*80)
        
        mlp_model = SimpleMLP(
            input_dim=X_train_scaled.shape[1],
            hidden_dims=[512, 256, 128],
            num_classes=len(self.label_encoder.classes_),
            dropout_rate=0.5
        ).to(self.device)
        
        mlp_model, mlp_train_accs, mlp_val_accs, mlp_train_losses, mlp_val_losses = self.train_model(
            mlp_model, 'MLP',
            X_train_part, y_train_part, X_val, y_val,
            epochs=200, batch_size=16, lr=0.001
        )
        
        mlp_results = self.evaluate_model(mlp_model, 'MLP', X_test_scaled, y_test)
        
        print(f"\n{'='*80}")
        print("🤖 MODEL 2: 1D CNN")
        print('='*80)
        
        cnn_model = GeneConv1D(
            input_dim=X_train_scaled.shape[1],
            num_classes=len(self.label_encoder.classes_),
            dropout_rate=0.5
        ).to(self.device)
        
        cnn_model, cnn_train_accs, cnn_val_accs, cnn_train_losses, cnn_val_losses = self.train_model(
            cnn_model, 'CNN',
            X_train_part, y_train_part, X_val, y_val,
            epochs=200, batch_size=16, lr=0.001
        )
        
        cnn_results = self.evaluate_model(cnn_model, 'CNN', X_test_scaled, y_test)
        
        print(f"\n{'='*80}")
        print("🤖 MODEL 3: AUTOENCODER + CLASSIFIER")
        print('='*80)
        
        ae_classifier = AutoencoderClassifier(
            autoencoder,
            encoding_dim=128,
            num_classes=len(self.label_encoder.classes_),
            dropout_rate=0.5
        ).to(self.device)
        
        ae_classifier, ae_train_accs, ae_val_accs, ae_train_losses, ae_val_losses = self.train_model(
            ae_classifier, 'AutoencoderClassifier',
            X_train_part, y_train_part, X_val, y_val,
            epochs=200, batch_size=16, lr=0.0005
        )
        
        ae_results = self.evaluate_model(ae_classifier, 'AutoencoderClassifier', X_test_scaled, y_test)
        
        training_curves = {
            'MLP': {
                'train_accs': mlp_train_accs,
                'val_accs': mlp_val_accs,
                'train_losses': mlp_train_losses,
                'val_losses': mlp_val_losses
            },
            'CNN': {
                'train_accs': cnn_train_accs,
                'val_accs': cnn_val_accs,
                'train_losses': cnn_train_losses,
                'val_losses': cnn_val_losses
            },
            'Autoencoder': {
                'train_accs': ae_train_accs,
                'val_accs': ae_val_accs,
                'train_losses': ae_train_losses,
                'val_losses': ae_val_losses
            }
        }
        
        self.plot_training_curves(training_curves)
        
        print(f"\n{'='*80}")
        print("📊 MODEL COMPARISON")
        print('='*80)
        
        comparison = pd.DataFrame([
            {'Model': 'MLP', 'Accuracy': mlp_results['accuracy'], 'AUC': mlp_results['auc']},
            {'Model': 'CNN', 'Accuracy': cnn_results['accuracy'], 'AUC': cnn_results['auc']},
            {'Model': 'Autoencoder+Classifier', 'Accuracy': ae_results['accuracy'], 'AUC': ae_results['auc']}
        ])
        
        print(comparison.to_string(index=False))
        comparison.to_csv(f'{self.output_dir}/model_comparison.csv', index=False)
        
        plt.figure(figsize=(10, 6))
        x = range(len(comparison))
        plt.bar(x, comparison['Accuracy'], alpha=0.7)
        plt.xticks(x, comparison['Model'])
        plt.ylabel('Accuracy')
        plt.title('Deep Learning Model Comparison')
        plt.ylim([0, 1])
        for i, v in enumerate(comparison['Accuracy']):
            plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/model_comparison.png', dpi=300)
        plt.close()
        
        print(f"\n{'='*80}")
        print("🔄 K-FOLD CROSS-VALIDATION")
        print('='*80)
        
        mlp_kfold, mlp_mean, mlp_std = self.run_kfold_evaluation(
            SimpleMLP,
            'MLP',
            {
                'input_dim': X_train_scaled.shape[1],
                'hidden_dims': [512, 256, 128],
                'num_classes': len(self.label_encoder.classes_),
                'dropout_rate': 0.5
            },
            n_splits=5
        )
        
        print("\n" + "="*80)
        print("✅ DEEP LEARNING PIPELINE COMPLETE!")
        print("="*80)
        print(f"\n📁 Results saved in: {self.output_dir}/")
        print("\nGenerated files:")
        print("   • MLP_confusion_matrix.png")
        print("   • CNN_confusion_matrix.png")
        print("   • AutoencoderClassifier_confusion_matrix.png")
        print("   • training_curves.png")
        print("   • model_comparison.png")
        print("   • model_comparison.csv")
        print("   • autoencoder_training.png")
        print("   • best_*.pth (model weights)")
        print("="*80)
        
        return {
            'mlp': mlp_results,
            'cnn': cnn_results,
            'autoencoder': ae_results,
            'comparison': comparison,
            'kfold_results': {
                'mlp': (mlp_kfold, mlp_mean, mlp_std)
            }
        }


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
    
    dl_pipeline = DeepLearningPipeline(expression_data, metadata)
    
    results = dl_pipeline.run_full_pipeline()
    
    print("\n" + "="*80)
    print("🎉 ALL DONE!")
    print("="*80)