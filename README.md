# UNADON
Transformer-based model to predict genome-wide chromosome spatial position

UNADON predicts the genome-wide cytological distance to a specific type of nuclear body measured by TSA-seq using both sequence features and epigenomic signals. The neural network architecture is described in the figure below.

![The overall architecture of UNADON](https://github.com/ma-compbio/UNADON/blob/main/Figure%201%20.png)

The major contributions of UNADON are as follows:
 -  UNADON is a deep learning-based model to specifically predict chromatin spatial positioning relative to nuclear bodies;
 -  The distinctive neural architecture design enables UNADON to learn the long-range dependencies more effectively;
 -  UNADON generalizes well in the cross-cell-type predictions, which can be applied to infer spatial positioning in new cell types.
 -  Interpretation of UNADON reveals potential mechanisms for targeting nuclear bodies.

# Requirements
UNADON is developed and tested under python 3.7.10.
List of required python packages for running UNADON:

 - bedtools=2.30.0
 - biopython=1.79
 - captum=0.5.0
 - h5py=3.1.0
 - numpy=1.19.5
 - pybedtools=0.8.2
 - pybigwig=0.3.18
 - scikit-learn=0.24.2
 - scipy=1.7.2
 - torch=1.12.1
 - torch-optimizer=0.3.0
 - torchvision=0.13.0
 - xgboost=1.5.0


# USAGE

The workflow of UNADON includes three steps: (1) Data preprocessing; (2) Model training and evaluation; (3) Model interpretation.

## Data preprocessing

To prepare the data for training UNADON, run the following command:

    python data_preprocessing.py

The configuration for data preprocessing is defined in config/config_data.json, which contains:

 - seq_path: the path to the genome sequences 
 - cell_type: the cell type for preprocessing
 - output_file: the path to the output hdf5 file
 - kmer_list: the list of k for k-mers
 - resolution: the size of the genomic bin
 - signal_path: the path to the epigenomic features and TSA-seq
 - epi_name: the names of the epigenomic features
 - centromere_path: the path to the centromere annotation
 - tsa_type: the list of TSA-seq targets (["SON", "LMNB"])
 - tsa_scaling: the scaling factor used for TSA-seq preprocessing


## Model training and evaluation

UNADON can be trained by running the following command:

    python train.py

The configuration for model training and evaluation is defined in config/config.json, which contains:

 - num_of_epochs: the maximum number of epochs for training
 - dropout: the dropout rate
 - base_lr: the base learning rate
 - batch: the batch size
 - reg: the parameter for weight decay in AdamW
 - window_size: the size of the context window
 - training_cell_type: the list of training cell types
 - validation_cell_type: the list of validation cell types
 - testing_cell_type: the list of testing cell types
 - mode: the type of experiment. "Single" for single-cell-type experiments, "Cross" for cross-cell-type experiments
 - run_test: whether to run the evaluation on the testing set. Should be False for hyperparameter tuning.
 - training_chr: the training chromosomes
 - validation_chr: the validation chromosomes
 - testing_chr: the testing chromosomes
 - train_data_path: the path to the processed data for training set
 - valid_data_path: the path to the processed data for validation set
 - test_data_path: the path to the processed data for testing set
 - feature: the input features to include (sequence features and epigenomic features)
 - y: the TSA-seq target  for prediction. Can be "SON" or "LMNB".
 - output_path: the path to the output directory
 - output_name: the name of the output files
 - histone: the list of ATAC-seq and histone features 
 - IML_cell_type: the cell type used for model interpretation
 - dense dim: the number of hidden units in the dense layer for the data processing subnetwork
 - dense_num_layers: the number of dense layers for the data processing subnetwork
 - nhead: the number of heads in the multi-head attention layer
 - attn_layers: the number of transformer encoder layers
 - attn_hidden_dim: the hidden dimension for the transformer encoder module
 - random_state: the random seed to ensure reproducibility
        

Besides, we have attached the architecture of the baseline models that we used for benchmarking, which can be found in baseline.py.

## Model interpretation

We have included the code for running Integrated Gradients to derive the feature importance score (model_interpretation.py) and performing k-means clusteringto identify the feature contribution patterns  (clustering.py)  as reference. 

