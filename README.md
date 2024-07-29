### TFIDF Fake News Detection
As the title suggests, the project focuses on exploring TFIDF vectorization and its potential.

In the first part, various data visualisation techniques, including WordCloud, are applied to understand the dataset and to provide an intuitive justification - supported 
by a mathematical description - for TFIDF vectorization, which is widely used in this type of task.

The second part is a straightforward application of Data Science methods: data preprocessing with `pandas`, `numpy` and `nltk`, model 
selection with `scikitlearn`, hyperparameter tuning with grid-search and cross-validation techniques. In this part, much importance is given to the running time and 
computational cost of the various models: the aim is to choose a simple, fast model with the highest possible performance.

The third and final part aims to answer the following question: is it possible to boost the model's performance by including the output of a pre-trained Sentiment Analysis model? 
To answer this, we embark on a journey into the world of news and the sentiments contained within them, trying to understand if "emotions" play a significant role in fake news 
detection. The final model is a neural network created with `keras` that takes as input feature vectors combining the output of the classifier 
trained on the texts and that of the Sentiment Analysis model.
