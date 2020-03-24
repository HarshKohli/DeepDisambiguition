# DeepDisambiguition

Various Deep Learning models to disambiguate between entities in sentences. Albert is used as a base embedding model. The methods used can extend to other tasks such as verb/word sense disambiguition/intent classification etc.

# Dependencies

tensorflow >= 2.0, pyyaml, transformers, scikit



# How to Run

python train_hard_negatives.py

python train_kl_divergence.py

python train_classifier.py

For 3 approaches - using hard negative sampling + triplet loss, optimizing embedding distributions using KL Divergence and N-class classification for N different entities in our corpus.

All necessary parameters can be set in config.yml

Sample data is provided in data/train.tsv . Any new data must conform to this schema

# TODOs

Model-Saving routines