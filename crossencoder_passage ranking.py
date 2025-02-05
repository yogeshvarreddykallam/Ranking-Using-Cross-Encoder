from torch import nn
import torch
from transformers import AutoModelForSequenceClassification
import torch.nn.functional as F

class CrossEncoder(nn.Module):
    def _init_(self, model_name_or_dir, dropout_rate=0.1) -> None:
        # Initialize the superclass (nn.Module)
        super()._init_()
        
        # Load a pre-trained transformer model for sequence classification
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_dir, num_labels=2
        )
        
        # Define the loss function (CrossEntropyLoss for binary classification)
        self.loss = nn.CrossEntropyLoss()
        
        # Define a dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout_rate)

    def score_pairs(self, pairs):
        # Forward pass through the transformer model
        outputs = self.model(**pairs, return_dict=True)
        logits = outputs.logits  # Get classification logits
        
        # Extract the [CLS] token embeddings from the last hidden state
        cls_embeddings = outputs.hidden_states[-1][:, 0, :]
        
        # Split embeddings into query and document pairs
        query_embeddings = cls_embeddings[::2]  # Every even index is a query
        doc_embeddings = cls_embeddings[1::2]   # Every odd index is a document
        
        # Compute cosine similarity between [CLS] token embeddings of query and document
        cosine_scores = F.cosine_similarity(query_embeddings, doc_embeddings)
        
        # Apply sigmoid activation to logits (to convert them into probability scores)
        scores = torch.sigmoid(logits[:, 1])
        
        return scores, cosine_scores

    def forward(self, pos_pairs, neg_pairs):
        # Compute scores for positive and negative pairs
        pos_scores, pos_cosine_scores = self.score_pairs(pos_pairs)
        neg_scores, neg_cosine_scores = self.score_pairs(neg_pairs)

        # Concatenate positive and negative scores
        scores = torch.cat([pos_scores, neg_scores], dim=0)
        cosine_scores = torch.cat([pos_cosine_scores, neg_cosine_scores], dim=0)

        # Apply dropout for regularization
        scores = self.dropout(scores)
        cosine_scores = self.dropout(cosine_scores)

        # Create labels (1 for positive pairs, 0 for negative pairs)
        pos_labels = torch.ones(pos_scores.size(), device=pos_scores.device)
        neg_labels = torch.zeros(neg_scores.size(), device=neg_scores.device)
        labels = torch.cat([pos_labels, neg_labels], dim=1)

        # Compute loss using both classification scores and cosine similarity scores
        loss = self.loss(scores, labels) + self.loss(cosine_scores, labels)

        return loss, scores, cosine_scores
