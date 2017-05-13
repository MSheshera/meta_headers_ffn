### Description:
Impliments a simple feed forward neural network for classifying tokens in
a research paper as being one of six types of metadata information. Other 
models which were also part of the project are here:

Mollys [BiLSTM](https://github.com/mmcmahon13/deep-metadata-extraction), 

Akul and Adityas [Logistic Regression baseline](https://github.com/akuls/crf-pdf-parsing)
    
### Dataset:
GROTOAP2: http://www.dlib.org/dlib/november14/tkaczyk/11tkaczyk.html

### Description of directories and files:
`feat_extract/*.py`: Code to extract features from the data stored as TrueViz xml files and serialize them to npy binary files. Extraction uses on Molly McMohans xml parsing code; also, extraction code is based on her feature extraction code for the BiLSTM which tries to solve the same problem.

`utils/*.py`: General IO utilities.

`/feat_extract/*.py`: Mollys xml parsing code; code to extract features from parsed xml and write to gigantic numpy matrices.

`settings.py`: Contains global settings.
    
`pre_proc.py`: Code to standardize extracted features.

`learning.py`: Code to learn the model. Model specified here.

`evaluate.py`: Code to evaluate the trained model(s).

`driver.py`: Runs pipeline and prints results to STDOUT.

### Running this code:
The following things need to happen:
* Create the simstring databases for use during feature extraction. Only need to do this one. (Run: `stringmatching.py`)
* Go over the xml files; extract features and write to a gigantic numpy feature matrix; this is the slowest step. Do this individually first for the train set and then the test set. (Run with appropriate argumets: `grotoap_to_npy.py -h`)
* Train and test a model etc. (Run with appropriate argumets: `driver.py -h`)

You mainly need to set output and input paths manually. Most other things happen
automatically. All output directories get created if they dont exist (exceptions mentioned below).

1. In `fe_settings.py`: Set the following variables:
 * `lexicon_dir`: to point to the directory with all the lexicon text files. 
 * `simstringdb_dir`: Path to the directory where you want the output simstring databases.
 * `grotoap_dir_train`: Top level GROTOAP train split directory. Should contain the numbered directories with the .cxml files in them. For example: ``grotoap_dir_train/00/*.cxml`` ``grotoap_dir_train/01/*.cxml`` etc.
 * `grotoap_dir_test`: Top level GROTOAP test split directory. Should contain the numbered directories with the .cxml files in them. For example: ``grotoap_dir_test/00/*.cxml`` ``grotoap_dir_test/01/*.cxml`` etc.
 * ``out_dir``: Directory to write extracted feature matrices (npy files) into.
 * ``map_dir``: Directories to which maps (what int value represents what label etc) should be written.
2. In `fe_settings.py`: Set the following variables:
 * `l_train_path`: Full name to train npy file. Extracted in feature extraction run.
 * `l_dev_path`: Full name to dev npy file. Extracted in feature extraction run.
 * `l_test_path`: Full name to test npy file. Extracted in feature extraction run.
 * `l_map_path`: Directories from which to read maps for the labels.
 * `l_check_dir`: Chcekpoint directory. Directory where models should be saved. Direcory not automatically created if doesnt exist.
 
Example run:

    # Create simstring databases.
    $ python stringmatching.py
    # Create train dataset and label int maps.
    $ python grotoap_to_npy.py -dataset train
    # Use int maps from above and create test set.
    $ python grotoap_to_npy.py -dataset test
    # Transform features to be standardized; train default model on local machine; evaluate model.
    $ python driver.py -t --machine local --action def_model
    # Transform features to be standardized; train tuned model on local machine; save the model.
    $ python driver.py -t --machine local --action tuned_model
    # Transform features to be standardized; load saved model on local machine; evaluate on test data.
    $ python driver.py -t --machine local --action named_model
    


### Note:
All of this was run on the Swarm2 cluster at UMass CICS and has a tonne of things hard coded to run there. Needs a tonne more work to be easy to run elsewhere.
