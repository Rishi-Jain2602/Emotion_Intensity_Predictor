# Emotion_Intensity_Predictor

***

## Requirements
- Python 3.x
- Jupyter Notebook
- Python Libraries - Numpy, sklearn, tensorflow , pandas , matplotlib , seaborn , nltk

***
 ## Installation
1. Clone the Repository
``` bash
git clone https://github.com/Rishi-Jain2602/Emotion_Intensity_Predictor.git
```
2. Install the Project dependencies
```bash
pip install -r requirements.txt
```
3. Install the Dataset 'Tweets_data.csv'
   
****

## Model Training

### Knowlege Required
- Natural Language Processing (NLP)
- Python Libraries - Numpy, sklearn, tensorflow , pandas , matplotlib , seaborn , nltk
- Machine Learning Algorithm  - SVM , Random Forest classifier , XGB Classifier

### Tools
- Vs code , Codelab

## Two models are made - Machine Learning Model (Purely Statistical Model) , Deep Learning Model

## Machine Learning Model (Model1)

### Data Cleaning
- Text is converted to lowercase
- Replace certain special characters with their string equivalent 
- Removing HTML tags and URLs
  
### Sentence Embeddings using Sentence Transformers

This code snippet demonstrates how to use the SentenceTransformer library to generate sentence embeddings for training and testing data. Sentence embeddings capture the semantic meaning of sentences and are useful for various Natural Language Processing (NLP) tasks.This sentence-transformers model: It maps sentences & paragraphs to a 768 dimensional dense vector space and can be used for tasks like clustering or semantic search.

#### Code Description:
- Import the SentenceTransformer library.
- Initialize a pre-trained model ('all-mpnet-base-v2') for generating sentence embeddings.
- Encode the training and testing data to obtain embeddings using the initialized model.

#### Library Used:
[Sentence Transformers(Hugging Face)](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
[Sentence Transformers(Github)](https://github.com/UKPLab/sentence-transformers)

#### Usage:
Ensure you have installed the Sentence Transformers library before running the code.

```
pip install -U sentence-transformers
```

### Model Used

- SVM - It's accuracy is 77.32%
- XGB Classifier- It's accuracy is 71.33%
- Random Forest Classifier is 67.88%
#### Classification Reports of the models

![r1](https://github.com/Rishi-Jain2602/Emotion_Intensity_Predictor/assets/118871883/511d1150-fd60-4ef6-9e10-773a9129f28c)


*SVM is giving better result in comparison to other two models*


#### Few Test Cases

![test cases1](https://github.com/Rishi-Jain2602/Emotion_Intensity_Predictor/assets/118871883/f19eea98-825b-4076-a235-2e7a8674272b)



***
## Deep Learning (Model2)

### Data Cleaning
- Text is converted to lowercase
- Replace certain special characters with their string equivalent and Decontracting words
- Removing HTML tags and URLs
- Removing Stopwords

### Tokenization and Padding with Tokenizer

This code snippet demonstrates how to use the `Tokenizer` utility provided by Keras for tokenizing and padding sequences of text data. Tokenization is the process of converting text into numerical sequences, while padding ensures uniform sequence lengths for input into neural networks.

#### Tokenizer Description:
- **Purpose**: Tokenizer is used to vectorize text data by converting words into integer indices.
- **OOV Token**: An out-of-vocabulary token (`oov_token`) is specified as `'nothing'`, representing words not in the vocabulary.
- **Fit on Texts**: `fit_on_texts()` updates the internal vocabulary based on training data (`X_train`).
- **Texts to Sequences**: `texts_to_sequences()` converts text data (`X_train` and `X_test`) into sequences of integers.
- **Padding**: Sequences are padded or truncated to a fixed length (`max_len`) using `pad_sequences()`, ensuring uniform input dimensions.

#### Usage:
Ensure you have installed the Keras library before running the code.
```
pip install tokenizer
```

### Sequential Model Architecture

This Sequential model consists of several layers, each serving a specific purpose in the neural network architecture.
#### Layers Description:
1. **Embedding Layer**:
   - Converts input text data into dense vectors of fixed size.
2. **Bidirectional LSTM Layer (First)**:
   - A Bidirectional LSTM layer with 30 units.
   - `return_sequences=True` ensures that the LSTM layer returns the full sequence of outputs for each input time step.
3. **Bidirectional LSTM Layer (Second)**:
   - Another Bidirectional LSTM layer with 30 units.
   - Since `return_sequences` is not specified, it defaults to `False`, meaning it returns only the output of the last timestep.
4. **Dense Layer (Output)**:
   - Output layer with 4 units and a softmax activation function.
   - The 4 units suggest a multi-class classification task with four output classes.

#### Model Compilation:
- The model is compiled using the Adam optimizer and sparse categorical crossentropy loss function, suitable for multi-class classification tasks.
- Accuracy is specified as the metric for evaluation during training.

#### Model Summary:
- The `model.summary()` function displays a summary of the model architecture, providing insights into the structure and parameters of each layer.
