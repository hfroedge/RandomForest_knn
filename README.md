# RandomForest_knn

**Lazy learning random forests with nearest neighbor localization for text classification**

**Culmination of work done by Seyma Nur Günel and Harrison Froedge. Started December 2020**

Pertinent code contained in random_forest.py and tuning.py. notebook.ipynb only contains some pre-processing work.

Random forest algorithms have seen considerable success since their emergence nearly two decades ago (Breiman, 2001). Random forests grow a number of decision trees in a random way (hence the name), selecting subsets of data attributes and sometimes bootstraps of training data to grow each component tree in a unique way. This method gives rise to an ensemble of better-than-random-guessing classifiers which, for a given test sample, each contribute their prediction (a "vote") for the final classification. The class which receives the majority of votes is chosen as the forest prediction
   
 Such constructs work together to reduce overfitting and increase generalization, areas which decision trees notoriously fall short. As noted by Salles et al. (2018), random forest classifiers reduce variance (as compared to single decision trees) while maintaining constant bias. This typically results in better, more robust models. Despite these gains over single trees, random forests still grapple with noise, and often times they succumb to the overly greedy and explicit fitting strategy of their component trees. This is especially a problem when the number of data attributes is very large or the data is sparse, since many random samples of attributes may create worthless trees which fit to some other, unrelated trend to the signal of interest. (Salles et al., 2018)
    
 To overcome the difficulties faced by random forests on noisy, high-dimensional data, authors Salles, T., Gonçalves, M., Rodrigues, V., & Rocha, L. presented the 2018 paper "Improving random forests by neighborhood projection for effective text classification", in which they detailed a novel lazy learning random forest, called LazyNN_RF, which trains itself only on the nearest neighbors to a test sample. By tuning the neighborhood of this projection (i.e., optimizing the number of neighbors), only samples of the same topic or nature need be considered. The authors claimed this implementation to significantly improve performance on certain noisy data sets over standard random forests and a number of other state-of-the-art classifiers. Furthermore, the authors claimed that such a procedure dramatically decreases training time for a random forest, since only a (usually very) small subset of data need be considered. The authors of this paper provided basic, high-level pseudocode for LazyNN_RF.
    
 Here, we present an implementation of LazyNN\_RF (which we rename RandomForest_knn).
    
    
## Intended Problem Domains
  
The authors of the subject paper (Salles et al., 2018) designed their LazyNN_RF especially for text-classification tasks. Text-classification is a domain replete with noise and subtle nuance, challenges this lazy-learner is meant to conquer (by localizing itself to a given input sample, thus ignoring irrelevant data).


## References

Breiman, L. (2001). Random Forests. _Machine Learning, 45_, 5-32. 
        
Salles, T., Gonçalves, M., Rodrigues, V., & Rocha, L. (2018). Improving random forests by neighborhood projection for effective text classification. _Information Systems, 77_, 1-21. https://doi.org/10.1016/j.is.2018.05.006
        
_Simulated Annealing_. Retrieved January 1, 2021, from https://santhoshhari.github.io/simulated\_annealing/
