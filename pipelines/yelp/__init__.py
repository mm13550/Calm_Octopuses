"""
pipelines/yelp/
===============
Training data preparation scripts for the Yelp Open Dataset MDN pipeline.

These scripts are run sequentially to build the training and validation datasets
used to train the ``MDNScorer`` (``algorithms/mdn_regression.py``).

Execution order
---------------
1. download_yelp_dataset.py       Download and extract the Kaggle Yelp dataset.
2. preprocess_yelp.py             Build the ``yelp_relations.db`` SQLite store.
3. generate_embeddings_yelp.py    Generate per-photo CLIP image + text embeddings.
4. aggregate_restaurant_embeddings.py  Pool embeddings to one vector per restaurant.
5. export_regression_train.py     Export the ``regression_train_set.json`` profile set.
6. mdn_regression.py (train)      Train and checkpoint the MDNScorer.
"""
