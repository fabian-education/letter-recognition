# Letter Recognition - Dataset Branch
This branch contains the training data and pre-trained models for the letter recognition AI.

## Structure
```
data/
├── datasets/
│   ├── big_dataset.zip      # Large training dataset
│   └── own_letters.zip      # Own training dataset 
├── raw_data/
│   └── letters.pdf          # Raw letter templates (private)
└── 57-60-bilder-*.jpg       # Scanned letter images (own)

all_models/
├── CNN_model.keras          # Trained CNN model
├── CNN_model_bigdata.keras  # CNN model trained on large dataset
├── feedforward_model.keras  # Feedforward neural network model
└── feedforward_model_bigdata.keras  # Feedforward model trained on large dataset
```

## Usage
Switch to the `master` branch for the main application code.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
