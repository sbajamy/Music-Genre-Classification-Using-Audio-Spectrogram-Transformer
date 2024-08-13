# Music-Genre-Classification-Using-Audio-Spectrogram-Transformer
### Fine tune an audio spectrogram transformer using GTZAN dataset to estimate music genre
![alt text](https://github.com/sbajamy/Music-Genre-Classification-Using-Audio-Spectrogram-Transformer/blob/main/images/OpeningImage.jpg)  
## Model  
AST: Audio Spectrogram Transformer Yuan Gong, Yu-An Chung, James Glass. The main idea is applying a visual transformer to the spectogram of a given audio signal in order to extract features for classification. The model was pretrained on AudioSet dataset which has a variety of labeled youtube audio signals(Labels include music,bark,engine etc.). [https://huggingface.co/docs/transformers/en/model_doc/audio-spectrogram-transformer]  
![alt text](https://github.com/sbajamy/Music-Genre-Classification-Using-Audio-Spectrogram-Transformer/blob/main/images/AST.jpg)  
## Dataset
"GTZAN is a dataset for musical genre classification of audio signals. The dataset consists of 1,000 audio tracks, each of 30 seconds long. It contains 10 genres, each represented by 100 tracks. The tracks are all 22,050Hz Mono 16-bit audio files in WAV format. The genres are: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock."[https://huggingface.co/datasets/marsyas/gtzan]
## Fine tuning method
Classic transfer learning scheme. Freezing all the pretrained AST model layers but the last layer. This layer is replaced with a new fully connected layer which is used to adapt AST to the new task during training.
## Results
Used 80% of the GTZAN samples as a training set and the rest were equally divided to a validation and test set(10% each).  
Using the section above's method during training and validation, the test set classification accuracy achieved was 85% and its confusion matrix:  
![alt text](https://github.com/sbajamy/Music-Genre-Classification-Using-Audio-Spectrogram-Transformer/blob/main/images/Test_confusion_matrix.jpg)   
Validation accuracy vs training iterations(Best model validation accuracy is 82%):  
![alt text](https://github.com/sbajamy/Music-Genre-Classification-Using-Audio-Spectrogram-Transformer/blob/main/images/Validation_vs_iterations.jpg)  
## Getting started
Use git to clone the repository with the following command:   
`git clone https://github.com/taldatech/ee046211-deep-learning.git`   
If an ece046211 virtual environment is already installed on your machine, activate it(`conda activate deep_learn`), skip to the transformers package installation in the table below and continue from there.    
Else:
1. Get Anaconda with Python 3, follow the instructions according to your OS (Windows/Mac/Linux) at: https://www.anaconda.com/download
2. Create a new environment for the course and install packages from scratch:
In Windows open `Anaconda Prompt` from the start menu, in Mac/Linux open the terminal and run `conda create --name deep_learn python=3.9`. Full guide at https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands
4. To activate the environment, open the terminal (or `Anaconda Prompt` in Windows) and run `conda activate deep_learn`
5. Install the required libraries according to the table below (to search for a specific library and the corresponding command you can also look at https://anaconda.org/)

### Libraries to Install

|Library         | Command to Run |
|----------------|---------|
|`Jupyter Notebook`|  `conda install -c conda-forge notebook`|
|`numpy`|  `conda install -c conda-forge numpy`|
|`matplotlib`|  `conda install -c conda-forge matplotlib`|
|`pandas`|  `conda install -c conda-forge pandas`|
|`scipy`| `conda install -c anaconda scipy `|
|`scikit-learn`|  `conda install -c conda-forge scikit-learn`|
|`seaborn`|  `conda install -c conda-forge seaborn`|
|`tqdm`| `conda install -c conda-forge tqdm`|
|`opencv`| `conda install -c conda-forge opencv`|
|`optuna`| `pip install optuna`|
|`pytorch` (cpu)| `conda install pytorch torchvision torchaudio cpuonly -c pytorch` (<a href="https://pytorch.org/get-started/locally/">get command from PyTorch.org</a>)|
|`pytorch` (gpu)| `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia` (<a href="https://pytorch.org/get-started/locally/">get command from PyTorch.org</a>)|
|`torchtext`| `conda install -c pytorch torchtext`|
|`torchdata`| `conda install -c pytorch torchdata` + `pip install portalocker`|
|`transformers`|`conda install -c conda-forge transformers`|
|`accelerate`|`conda install -c conda-forge accelerate`|
|`datasets`|`conda install -c conda-forge datasets`|
|`evaluate`|`conda install -c conda-forge evaluate`|
|`pydub`|`conda install -c conda-forge pydub`|
|`audiomentations`|`pip install audiomentations`|
|`librosa`|`conda install -c conda-forge librosa`|
|`tensorboardX`|`conda install -c conda-forge tensorboardX`|
## Train and Test Notebooks
There are two jupyter notebooks in the repository:  
* `train_test_gtzan.ipynb`- Trains the AST model on GTZAN dataset using the suggested fine tuning method, save the most accurate model on the validation set and show the test set results of it.
* `test_best_model.ipynb`- Test the current saved best music genre classification model capabilities on your own music files.  

To open a notebook, open Ananconda Navigator or run `jupyter notebook` in the terminal (or `Anaconda Prompt` in Windows) while the `deep_learn` environment is activated.   
## References
* Yuan Gong and Yu-An Chung and James Glass "AST: Audio Spectrogram Transformer", 2021,Proc. Interspeech 2021,571-575.
* https://huggingface.co/docs/transformers/en/model_doc/audio-spectrogram-transformer
* https://huggingface.co/learn/audio-course/en/chapter4/fine-tuning#conclusion
* https://huggingface.co/datasets/marsyas/gtzan