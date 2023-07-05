# Author Name Disambiguation for Scientific publications using Machine Learning, and Community Detection.


[src](src) contains the source code for the tasks.

 [src/preprocess.py](src/preprocess.py) contains the preprocessing methods for data extraction such
as extracting all article titles, article title per paper, extracting authors of articles, and
all authors in the references. It also contains the implementation of author disambiguation by 
 grouping.

[src/models.py](src/models.py) has machine learning models implementation for classification.

[src/evaluate.py](src/evaluate.py) has evaluation methods.

[src/train.py](src/train.py) has training methods for classification and can be run as

``python train.py``

[src/batch_score.py](src/batch_score.py) has classifier prediction methods.

Before creating the environment, please unzip the data files.
#### Creating environment
```
python -m venv disambiguation
source disambiguation/bin/activate
pip install -r requirements.txt
```

### Data Preprocessing
Extracting all article titles, article title per paper, extracting authors of articles, and
all authors in the references from lxml files by parsing. It also contains the implementation of
creating author dictionary which maps to article id, publication, affiliation. I have also
implemented a method to count the citations of authors in references as ref author:citation dictionary. It can be found
in [Notebooks/Data_Processing_disambiguation_by_grouping.ipynb](Notebooks/Data_Processing_disambiguation_by_grouping.ipynb)

### Data Cleaning 

on **Article Title, Journal Title, Journal Subject, Affiliations**: Stopwords (only English), numeric, punctuations, etc.
were removed. (Stemming can also be done)

For **author names**: Dr., Prof., Prof.Dr., Jr., Sr., PhD., MD. were removed for standardizing author names 
and were converted into lower case.

## Author Disambiguation Methods:
** **
### 1. Grouping by author features
Author features can be used such as affiliation, contact details, publication, research areas, co-authors to disambiguate authors
with the same name. There can be multiple authors with the same name but in different institutions, different research
areas, etc. 

I have implemented this approach and disambiguated the authors with the same name by article id, article title,
affiliation, journal title, journal subject. Then this disambiguated data was used for classification approach. 
The implementation of author disambiguation by grouping can be found here  [src/preprocess.py](src/preprocess.py), 


If two authors with the same name have same affiliation, email id, and overlapping co-authors then 
pairwise similarity matching on strings can be considered or algorithm such as Edit-distance on authors name 
can also be used to group same identical author with different name representation and initials such as M. Niyogi,
Niyogi, M., Niyogi Mitodru, etc.

### 2.Classification Approach: 

After preparing the disambiguated dataset after scraping from lxml files to a csv file. 
Given the author names, affiliations, publication, journal, research area, venue, the problem can be treated as
multi-class classification to predict the author given publication. In this case, same author names with different 
affiliations are already being treated as disambiguated. As the dataset is very small for training, validation,
and testing, the classifier is unable to learn from the limited training data as there are more than 100 authors without
considering referenced authors but just 50 publications. 

| Authors | Article ID | Article Title | Journal Title | Journal Subject | Affiliation |
|---------|------------|---------------|---------------|-----------------|-------------|

TF-IDF vectorization was created to build publication features matrix and then these feature was fed into machine
learning algorithms such as Decision Tree, Random Forest, Multinomial Naive Bayes, etc.

Repeated Stratified Cross validation and evaluation metrics such as precision, recall, F1-score have been
considered. The implementation and experiments are available in [Notebooks/Author-disam-Classification.ipynb](Notebooks/Author-disam-Classification.ipynb)

In the future, I would like to use Word2Vec, Glove, BERT embeddings by training on scientific articles to create contextual embeddings
of scientific jargon.
### 3. Clustering Approach

Clustering algorithms like KMeans, Hierarchical, Agglomerative Clustering, density based clustering methods
can be used for this problem. The author features can be clustered and similar author features are expected to be 
grouped together in the clusters.

I have also implemented clustering approach. Features like authors, article title, and references are considered and 
they were clustered using KMeans and Agglomerative Clustering algorithms. The clusters have been 
visualized by TSNE. The experiments can be found here  [Notebooks/Author-disam-clustering.ipynb](Notebooks/Author-disam-clustering.ipynb)

### 4. Network community detection

Construct a co-authorship network:

Create a graph where each node represents an author and an edge exists between two nodes if the corresponding authors
have co-authored a paper together.
The weight of the edge can be set to the number of papers co-authored by the two authors.

Building a network graph between author and co-authors Community detection algorithm such as Louvain to partition
the graph into clusters where authors in the same cluster are more likely to be the same person.can be used to 
detect unique communities. Further information such as author affiliations, publication venues, and keywords
can be used to resolve any remaining ambiguities. 
The experiments using Louvain algorithm can be found here  [Notebooks/Author-disam-clustering.ipynb](Notebooks/Author-disam-clustering.ipynb)

### 5. Deep learning and graph neural network approaches 
In the future, I would like to try out SOTA architecture for author name disambiguation task.
