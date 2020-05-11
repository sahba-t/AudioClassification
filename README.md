# AudioClassification
***CS 529 Spring 20, Project 3 (Audio categorization)***<br>
***Members***: Sahba Tashakkori, Mauricio Monsivais <br>
***Team name on Kaggle***: Team GPGPU<br>
## Folder Structure
* src: All the source code for the project</li>
  * ***algorithm.ipynb*** a jupyter notebook conaining neural network and decision tree with PCA and domain features
  * FFNN.py: Our experimentations on Xena, mostly same functions in the notebook + some utility functions.
  * pca_ops.py: all the methods necessary to read wave files, apply pca, and saved the result
  * mp3_2_wav.py: The script to convert mp3 files to wav files
  * ***CNN.ipynb*** reads wav files, writes spectrograms, reads spectrograms, trains and predicts sprectrograms using with CNN
* res: the files needed for the algorithm including feature csv files, spectograms, training labels, ...</li>
* kaggle_submissions: the kaggle submission files that signify major improvements in our project
