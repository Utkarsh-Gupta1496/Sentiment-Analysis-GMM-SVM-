# Sentiment Analysis


**Dataset Details :** (./data/movieReviews1000.txt)
1. Movie Review data (each line is a individual review)
2. Each Review has a label associated with it either 1 (Positive Review) or 0 (Negative Review). Label is mentioned at the end of each review.

**Features :** 
1. We extract TF-IDF feature for each Review
2. Reduce the TF-IDF feature dimmension to 10 using PCA
3. Split Reviews for Training (700 Reviews) and Testing (300 Reviews)

# Unsupervised Sentiment Analysis using GMM

**Implementation:** Implemented GMM using EM from Scratch in Numpy.

**Training:**\
Gaussian Mixture Model is an unsupervised clustering technique and does not require label information. Here we train a **Two Mixture Diagonal Covariance GMM** on this data.
Progress of **Expectation Maximization(EM) Algorithm**can be observed by observing plot of log-liklihood vs iteration.

**Testing:**
1. For Testing we take a particular review(Say labled 1 review)
2. We find the liklihood(posterior probablity) of the test point using parameter of each gaussian fitted(one gaussian is fitted for label 1 review and other fitted for label 0 reviews)
3. Then on comparing the two obtained liklihood we assign the test point to the class(label1 or label 0) whose liklihood is greator.
4. Classification Accuracy = [(Number of Reviews Correctly Classified)/(Total Number of Reviews)]*100

**Sample Output(GMM)** :

**EM Iteration = 1**
![](/images/output.png)

**EM Iteration = 16 (Center of each cluster shifts when compared to iteration 1 inorder to maximize log liklihood)**
![](/images/output1.PNG)

**EM Algorithm Progress - Log Liklihood vs Iteration**
![](/images/output2.PNG)

**Classification Accuracy**
![](/images/output3.png)
# Supervised Sentiment Analysis using SVM:

Library Used for Implementing : LIBSVM

**Classification Accuracy**
![](/images/output4.PNG)





