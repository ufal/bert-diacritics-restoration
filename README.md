# Diacritics Restoration using BERT with Analysis on Czech language

We propose a new architecture for diacritics restoration based on contextualized embed-dings, namely BERT, and we evaluate it on 12 languages with diacritics. Furthermore, we con-duct a detailed error analysis on Czech, a morphologically rich language with a high level ofdiacritization. Notably, we manually annotate all mispredictions, showing that roughly 44% ofthem are actually not errors, but either plausible variants (19%), or the system corrections oferroneous data (25%). Finally, we categorize the real errors in detail.

## Requirements:

The packages we used are listed in requirements.txt. We use Python 3.7.

## Training Model

To train a model, follow these steps:

1. obtain a dataset (dataset for 12 languages can be downloaded from https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2607)
2. generate subword frequencies (these are later used for computing target instruction set)
```
python generate_subword_frequencies.py input_file_without_diacritics.txt target_file_with_diacritics.txt
```
3. train model

See ```run_training.sh```.

4. use trained model to generate diacritics

See ```run_predict.sh```.

## Best Predictions

We release the best predictions of our model on dataset from https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-2607). The predictions are in [predictions](predictions)

## Additional Czech Data

We tested the Czech model in other domains. The testing data alongside binary evaluation masks and our predictions are in [other_domain_evaluation](other_domain_evaluation). 

## Acknowledgements

## References