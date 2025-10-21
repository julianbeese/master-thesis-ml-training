"""
Data Loader für DuckDB Training Data
Lädt Daten aus der DuckDB-Datenbank und bereitet sie für das Training vor.
"""
import duckdb
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import yaml
from torch.utils.data import Dataset
import torch


class DebateDataset(Dataset):
    """PyTorch Dataset für Brexit Debate Daten"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 2048):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenisierung
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)  # Long für Single-Label
        }


class DuckDBDataLoader:
    """Lädt Daten aus DuckDB und erstellt PyTorch Datasets"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.db_path = Path(config_path).parent.parent / self.config['data']['database_path']
        self.table_name = self.config['data']['table_name']
        self.text_column = self.config['data']['text_column']
        self.label_column = self.config['data']['label_column']
        self.split_column = self.config['data']['split_column']
        self.max_length = self.config['data']['max_length']
        self.label2id = self.config['data']['label2id']
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.num_labels = self.config['data']['num_labels']
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Lädt Train und Test Daten aus DuckDB"""
        print(f"Lade Daten aus {self.db_path}...")
        
        conn = duckdb.connect(str(self.db_path), read_only=True)
        
        # Train Daten laden
        train_query = f"""
        SELECT * FROM {self.table_name}
        WHERE {self.split_column} = '{self.config['data']['train_split']}'
        """
        train_df = conn.execute(train_query).fetchdf()
        
        # Test Daten laden
        test_query = f"""
        SELECT * FROM {self.table_name}
        WHERE {self.split_column} = '{self.config['data']['test_split']}'
        """
        test_df = conn.execute(test_query).fetchdf()
        
        conn.close()
        
        print(f"Train Samples: {len(train_df)}")
        print(f"Test Samples: {len(test_df)}")
        
        return train_df, test_df
    
    def encode_labels(self, labels: pd.Series) -> List[int]:
        """Konvertiert Label-Strings zu IDs"""
        return [self.label2id.get(label, self.label2id["None"]) for label in labels]
    
    def prepare_datasets(self, tokenizer) -> Tuple[DebateDataset, DebateDataset]:
        """Bereitet Train und Test Datasets vor"""
        train_df, test_df = self.load_data()
        
        # Train Dataset
        train_texts = train_df[self.text_column].tolist()
        train_labels = self.encode_labels(train_df[self.label_column])
        
        # Test Dataset
        test_texts = test_df[self.text_column].tolist()
        test_labels = self.encode_labels(test_df[self.label_column])
        
        train_dataset = DebateDataset(
            train_texts, 
            train_labels, 
            tokenizer, 
            self.max_length
        )
        
        test_dataset = DebateDataset(
            test_texts, 
            test_labels, 
            tokenizer, 
            self.max_length
        )
        
        return train_dataset, test_dataset
    
    def get_label_info(self) -> Dict:
        """Gibt Informationen über die Labels zurück"""
        train_df, _ = self.load_data()
        
        info = {
            'num_labels': self.num_labels,
            'label_names': list(self.label2id.keys()),
            'label2id': self.label2id,
            'id2label': self.id2label,
            'label_distribution': train_df[self.label_column].value_counts().to_dict()
        }
        
        return info


def main():
    """Test Funktion"""
    from transformers import AutoTokenizer
    
    config_path = Path(__file__).parent.parent / "config" / "training_config.yaml"
    
    loader = DuckDBDataLoader(config_path)
    
    # Label Info anzeigen
    label_info = loader.get_label_info()
    print("\nLabel Information:")
    print(f"Anzahl Labels: {label_info['num_labels']}")
    print(f"Label Namen: {label_info['label_names']}")
    
    # Beispiel: Dataset erstellen (benötigt Tokenizer)
    # tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt-oss-20b")
    # train_dataset, test_dataset = loader.prepare_datasets(tokenizer)
    # print(f"\nDataset erstellt: {len(train_dataset)} train, {len(test_dataset)} test")


if __name__ == "__main__":
    main()
