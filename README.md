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

`data_utils/feature_extract.py`: Windowing on time series and extraction of features.

`settings.py`: Contains global settings.
    
`pre_proc.py`: Code to standardize extracted features.

`learning.py`: Code to learn the model. Model specified here.

`evaluate.py`: Code to evaluate the trained model(s).

`driver.py`: Runs pipeline and prints results to STDOUT.

### Note:
All of this was run on the Swarm2 cluster at UMass CICS and has a tonne of things hard coded to run there. Needs a tonne more work to be easy to run elsewhere.
