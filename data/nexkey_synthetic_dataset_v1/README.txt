NexKey-style Synthetic Deal Matching Dataset (v1)

Files:
- properties.csv: 15,000 synthetic deals/properties
- queries.csv: 30,000 synthetic buyer prompts + structured filters
- interactions.csv: 480,000 query-property pairs with relevance labels (0-3)

Use-case:
Train a recommender / retrieval model (e.g., dual-encoder or cross-encoder) in PyTorch:
- Input: query_text
- Candidate docs: property text built from the property fields
- Supervision: relevance (0-3) or match_score (continuous)

Notes:
- This is fully synthetic data (no MLS/private listing data).
- Locations are plausible but not real addresses of actual listings.
- Labels are generated from a rule-based scorer; treat them as 'weak supervision' and consider fine-tuning with real user feedback.
