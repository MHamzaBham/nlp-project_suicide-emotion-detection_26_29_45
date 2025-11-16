
*Dataset Source*: [https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)

# NLP Project: Suicide Detection

A machine learning project to detect suicidal intent from text using NLP. The model is trained in a Jupyter notebook and deployed via a simple Flask web app.


### File Descriptions

- **`main.ipynb`**  
  - Loads and preprocesses the dataset from `data/`.  
  - Trains an NLP model (e.g., using scikit-learn, TensorFlow, or Hugging Face).  
  - Saves the trained model to the `models/` folder.

- **`app.py`**  
  - A Flask web server that:  
    - Serves web pages (from `templates/`).  
    - Accepts user input (text).  
    - Loads the trained model from `models/`.  
    - Returns prediction (suicide risk: Yes/No or probability).
