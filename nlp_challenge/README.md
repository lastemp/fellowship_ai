# nlp_challenge

This is an AI program that performs Sentiment analysis on IMDB Dataset of 50K Movie Reviews.
It forms part of the NLP Challenges for the [Fellowship Program](https://www.fellowship.ai/).

The program has below listed dependency:
- [Scikit-learn](https://scikit-learn.org/stable/) is an open source machine learning library that supports supervised and unsupervised learning.
- [pandas](https://pandas.pydata.org/) is an open source, BSD-licensed library providing high-performance, easy-to-use data structures and data analysis tools for the Python programming language..

## Usage

All the following commands assume that your current working directory is _this_ directory. I.e.:

```console
$ pwd
.../nlp_challenge
```

1. Install the required Python packages for the project:

   ```sh
   pip3 install -r requirements.txt
   ```
   
1. Run the application:

   ```sh
   py nlp_challenge.py aclImdb\train aclImdb\test
   ```