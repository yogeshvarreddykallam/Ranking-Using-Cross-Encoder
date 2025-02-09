This code defines a custom PyTorch model class `CrossEncoder` that uses a pre-trained transformer model for sequence classification (e.g., BERT, RoBERTa, etc.) and is adapted for scoring pairs of sentences. The primary aim of this model is to compute both the classification score (binary classification) and the cosine similarity score between query-document pairs, which could be useful in tasks like **information retrieval**, **ranking**, or **text similarity**. Here's a detailed breakdown of each part of the class:

### 1. **Initialization (`__init__` method)**:
   ```python
   def __init__(self, model_name_or_dir, dropout_rate=0.1) -> None:
       super().__init__()
       self.model = AutoModelForSequenceClassification.from_pretrained(
           model_name_or_dir, num_labels=2
       )
       self.loss = nn.CrossEntropyLoss()
       self.dropout = nn.Dropout(dropout_rate)
   ```
   - **`model_name_or_dir`**: The path or name of the pre-trained transformer model to load (e.g., `"bert-base-uncased"`).
   - **`self.model`**: This loads a pre-trained transformer model for sequence classification from the `transformers` library. The model is configured to have 2 output labels (for binary classification).
   - **`self.loss`**: A CrossEntropyLoss criterion is used to calculate the classification loss (binary classification).
   - **`self.dropout`**: Dropout layer for regularization (preventing overfitting). It randomly disables a fraction of units during training.

### 2. **Scoring Pairs (`score_pairs` method)**:
   ```python
   def score_pairs(self, pairs):
       outputs = self.model(**pairs, return_dict=True)
       logits = outputs.logits
       cls_embeddings = outputs.hidden_states[-1][:, 0, :]
       query_embeddings = cls_embeddings[::2]
       doc_embeddings = cls_embeddings[1::2]
       cosine_scores = F.cosine_similarity(query_embeddings, doc_embeddings)
       scores = torch.sigmoid(logits[:, 1])
       return scores, cosine_scores
   ```
   This method takes a batch of pairs (`pairs`), where each pair consists of a query and a document. The goal is to return two types of scores: 
   - **Classification scores (`scores`)**: Whether the query and document pair is a positive match (1) or negative match (0).
   - **Cosine similarity scores (`cosine_scores`)**: A measure of semantic similarity between the query and document.
   
   **Steps**:
   1. **`outputs = self.model(**pairs)`**: Passes the `pairs` through the transformer model to get the outputs.
   2. **`logits = outputs.logits`**: Extracts the classification logits (raw prediction values before applying activation).
   3. **`cls_embeddings = outputs.hidden_states[-1][:, 0, :]`**: Extracts the embeddings of the `[CLS]` token from the final layer of the transformer. The `[CLS]` token is typically used to represent the overall sentence meaning.
   4. **`query_embeddings` and `doc_embeddings`**: Splits the embeddings into query and document embeddings (assuming pairs are arranged such that even indices correspond to queries and odd indices correspond to documents).
   5. **`cosine_scores = F.cosine_similarity(query_embeddings, doc_embeddings)`**: Calculates the cosine similarity between the query and document embeddings.
   6. **`scores = torch.sigmoid(logits[:, 1])`**: Applies a sigmoid function to the logits to get the probability that the pair is positive (binary classification).
   
   The method returns both the classification scores and cosine similarity scores.

### 3. **Forward Pass (`forward` method)**:
   ```python
   def forward(self, pos_pairs, neg_pairs):
       pos_scores, pos_cosine_scores = self.score_pairs(pos_pairs)
       neg_scores, neg_cosine_scores = self.score_pairs(neg_pairs)
       scores = torch.cat([pos_scores, neg_scores], dim=0)
       cosine_scores = torch.cat([pos_cosine_scores, neg_cosine_scores], dim=0)
       scores = self.dropout(scores)
       cosine_scores = self.dropout(cosine_scores)
       pos_labels = torch.ones(pos_scores.size(), device=pos_scores.device)
       neg_labels = torch.zeros(neg_scores.size(), device=neg_scores.device)
       labels = torch.cat([pos_labels, neg_labels], dim=1)
       loss = self.loss(scores, labels) + self.loss(cosine_scores, labels)
       return loss, scores, cosine_scores
   ```
   This method takes in two sets of sentence pairs:
   - **`pos_pairs`**: Positive pairs (query, document) that should be semantically related.
   - **`neg_pairs`**: Negative pairs (query, document) that are unrelated or irrelevant.

   **Steps**:
   1. **`pos_scores, pos_cosine_scores = self.score_pairs(pos_pairs)`**: Computes scores for the positive pairs using the `score_pairs` method.
   2. **`neg_scores, neg_cosine_scores = self.score_pairs(neg_pairs)`**: Computes scores for the negative pairs.
   3. **Concatenate scores**: Combines both the positive and negative scores for classification and cosine similarity.
   4. **Apply dropout**: Applies dropout regularization to the concatenated scores.
   5. **Create labels**: Labels for positive pairs are 1 (`pos_labels`), and for negative pairs, labels are 0 (`neg_labels`).
   6. **`loss = self.loss(scores, labels) + self.loss(cosine_scores, labels)`**: Computes the total loss by adding:
      - The CrossEntropy loss for the classification scores (`scores`).
      - The CrossEntropy loss for the cosine similarity scores (`cosine_scores`).

   The method returns the total loss and the individual scores (classification and cosine similarity).

### Summary:
This model, `CrossEncoder`, is designed for scoring pairs of sentences in tasks like sentence matching or ranking. It leverages a pre-trained transformer model for classification and computes two scores:
1. **Binary classification score** for whether the query and document are a match (positive or negative).
2. **Cosine similarity** between the query and document to capture semantic similarity.

The model uses dropout for regularization and computes the loss based on both the classification scores and cosine similarity, encouraging the model to optimize for both types of measurements.
