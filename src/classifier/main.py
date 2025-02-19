import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional 
import logging
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.decomposition import LatentDirichletAllocation
from bertopic import BERTopic
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
import json
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TopicModeler:
    """Handles the unsupervised topic modeling part of the pipeline."""
    
    def __init__(self, method: str = 'bertopic'):
        """
        Initialize the topic modeler.
        
        Args:
            method: Either 'lda' or 'bertopic'
        """
        self.method = method
        self.model = None
        self.vectorizer = None
        
    def train(self, texts: List[str], n_topics: int = 10) -> Dict:
        """
        Train the topic model and return discovered topics.
        """
        if self.method == 'lda':
            return self._train_lda(texts, n_topics)
        else:
            return self._train_bertopic(texts, n_topics)
    
    def _train_lda(self, texts: List[str], n_topics: int) -> Dict:
        """Train LDA model."""
        # Create and fit vectorizer
        self.vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        doc_term_matrix = self.vectorizer.fit_transform(texts)
        
        # Train LDA
        self.model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            learning_method='batch'
        )
        self.model.fit(doc_term_matrix)
        
        # Get top words for each topic
        feature_names = self.vectorizer.get_feature_names_out()
        topics = {}
        for topic_idx, topic in enumerate(self.model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
            topics[f"Topic_{topic_idx}"] = top_words
            
        return topics
    
    def _train_bertopic(self, texts: List[str], n_topics: int) -> Dict:
        """Train BERTopic model."""
        # Initialize and train BERTopic
        self.model = BERTopic(nr_topics=n_topics)
        topics, _ = self.model.fit_transform(texts)
        
        # Get topic information
        topic_info = self.model.get_topic_info()
        topics = {}
        for _, row in topic_info.iterrows():
            if row['Topic'] != -1:  # Skip outlier topic
                topics[f"Topic_{row['Topic']}"] = self.model.get_topic(row['Topic'])
                
        return topics

class TaxonomyManager:
    """Manages the creation and refinement of the taxonomy."""
    
    def __init__(self, taxonomy_path: Optional[Path] = None):
        self.taxonomy_path = taxonomy_path
        self.taxonomy = self._load_taxonomy() if taxonomy_path else {}
        
    def create_initial_taxonomy(self, topics: Dict) -> Dict:
        """
        Create initial taxonomy from topic modeling results.
        """
        taxonomy = {
            "categories": {},
            "metadata": {
                "version": "1.0",
                "created_from": "topic_modeling",
                "last_updated": pd.Timestamp.now().isoformat()
            }
        }
        
        for topic_id, words in topics.items():
            taxonomy["categories"][topic_id] = {
                "name": f"Category_{topic_id}",
                "keywords": words,
                "description": "Auto-generated category from topic modeling",
                "subcategories": {}
            }
            
        self.taxonomy = taxonomy
        self._save_taxonomy()
        return taxonomy
    
    def update_taxonomy(self, updates: Dict) -> Dict:
        """
        Update taxonomy with expert refinements.
        """
        self.taxonomy.update(updates)
        self.taxonomy["metadata"]["last_updated"] = pd.Timestamp.now().isoformat()
        self._save_taxonomy()
        return self.taxonomy
    
    def _save_taxonomy(self):
        """Save taxonomy to file."""
        if self.taxonomy_path:
            with open(self.taxonomy_path, 'w') as f:
                json.dump(self.taxonomy, f, indent=2)
                
    def _load_taxonomy(self) -> Dict:
        """Load taxonomy from file."""
        if self.taxonomy_path and self.taxonomy_path.exists():
            with open(self.taxonomy_path) as f:
                return json.load(f)
        return {}

class SupportCallDataset(Dataset):
    """PyTorch dataset for support call data."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class SupervisedClassifier:
    """Handles the supervised classification part of the pipeline."""
    
    def __init__(self, model_name: str = 'bert-base-uncased'):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
    def train(
        self,
        texts: List[str],
        labels: List[int],
        num_epochs: int = 3,
        batch_size: int = 16
    ) -> Dict:
        """
        Train the supervised classifier.
        """
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(set(labels))
        )
        
        # Create dataset and split
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        train_dataset = SupportCallDataset(X_train, y_train, self.tokenizer)
        val_dataset = SupportCallDataset(X_val, y_val, self.tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Training loop
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                train_loss += loss.item()
                
                loss.backward()
                optimizer.step()
            
            # Validation
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    val_loss += outputs.loss.item()
                    
                    predictions = torch.argmax(outputs.logits, dim=1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
            
            val_accuracy = correct / total
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"Training Loss: {train_loss/len(train_loader):.4f}")
            logger.info(f"Validation Loss: {val_loss/len(val_loader):.4f}")
            logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
            
        return {
            "final_val_accuracy": val_accuracy,
            "final_val_loss": val_loss/len(val_loader)
        }
    
    def predict(self, texts: List[str]) -> List[int]:
        """
        Predict categories for new texts.
        """
        dataset = SupportCallDataset(texts, [0]*len(texts), self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=16)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()
        
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                batch_preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(batch_preds.cpu().numpy())
                
        return predictions

class HybridClassifier:
    """Main class that orchestrates the hybrid classification approach."""
    
    def __init__(
        self,
        topic_method: str = 'bertopic',
        model_name: str = 'bert-base-uncased',
        taxonomy_path: Optional[Path] = None
    ):
        self.topic_modeler = TopicModeler(method=topic_method)
        self.taxonomy_manager = TaxonomyManager(taxonomy_path)
        self.supervised_classifier = SupervisedClassifier(model_name=model_name)
        
    def discover_topics(self, texts: List[str], n_topics: int = 10) -> Dict:
        """
        Perform initial topic modeling.
        """
        topics = self.topic_modeler.train(texts, n_topics)
        taxonomy = self.taxonomy_manager.create_initial_taxonomy(topics)
        return taxonomy
    
    def update_taxonomy(self, updates: Dict) -> Dict:
        """
        Update taxonomy with expert refinements.
        """
        return self.taxonomy_manager.update_taxonomy(updates)
    
    def train_classifier(
        self,
        texts: List[str],
        labels: List[int],
        num_epochs: int = 3
    ) -> Dict:
        """
        Train the supervised classifier.
        """
        return self.supervised_classifier.train(texts, labels, num_epochs)
    
    def predict(self, texts: List[str]) -> List[int]:
        """
        Predict categories for new texts.
        """
        return self.supervised_classifier.predict(texts)
    
    def save_model(self, path: Path):
        """Save the trained model and related artifacts."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save the supervised model and tokenizer
        self.supervised_classifier.model.save_pretrained(path / 'supervised_model')
        self.supervised_classifier.tokenizer.save_pretrained(path / 'tokenizer')
        
        # Save the topic modeler
        with open(path / 'topic_modeler.pkl', 'wb') as f:
            pickle.dump(self.topic_modeler, f)
    
    @classmethod
    def load_model(cls, path: Path):
        """Load a saved model."""
        path = Path(path)
        
        # Create instance
        instance = cls()
        
        # Load supervised model and tokenizer
        instance.supervised_classifier.model = (
            AutoModelForSequenceClassification.from_pretrained(path / 'supervised_model')
        )
        instance.supervised_classifier.tokenizer = (
            AutoTokenizer.from_pretrained(path / 'tokenizer')
        )
        
        # Load topic modeler
        with open(path / 'topic_modeler.pkl', 'rb') as f:
            instance.topic_modeler = pickle.load(f)
            
        return instance


def main():
    """Example usage of the hybrid classifier."""
    # Initialize classifier
    classifier = HybridClassifier(
        topic_method='bertopic',
        taxonomy_path=Path('taxonomy.json')
    )
    
    # Load processed transcriptions
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / 'processed_transcriptions'
    texts = []
    for file_path in processed_dir.glob('*.txt'):
        with open(file_path) as f:
            texts.append(f.read())
    
    # 1. Discover initial topics
    taxonomy = classifier.discover_topics(texts, n_topics=10)
    print("\nDiscovered Topics:")
    print(json.dumps(taxonomy, indent=2))
    
    # 2. At this point, you would manually review and refine the taxonomy
    # 3. Create labeled dataset (manual step)
    # 4. Train supervised classifier (once you have labels)
    # labels = [0, 1, 2, ...]  # Your manual labels
    # results = classifier.train_classifier(texts, labels)
    
    # 5. Save the model
    # classifier.save_model(Path('saved_model'))

if __name__ == "__main__":
    main()
