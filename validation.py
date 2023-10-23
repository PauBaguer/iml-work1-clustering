import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, davies_bouldin_score, silhouette_score
from sklearn.metrics.cluster import contingency_matrix, adjusted_rand_score, normalized_mutual_info_score, homogeneity_score
import fcmeans, pam, kmodes, kmeans, clarans

class validation():
    def __init__(self, algorithm, data, labels, v, k):
        self.algorithm = algorithm
        self.data = data
        self.labels = labels
        self.v = v
        self.k = k
    
    def csearch(self, crange, *args): # get the optimal number of clusters
        scores = []
        for n_clusters in range(2,crange+1):
            if self.algorithm == fcmeans.fcm:
                out = self.algorithm(self.data, n_clusters)
                labels = out[0].argmax(axis=1)
            elif self.algorithm == pam.pam:   
                data = pd.DataFrame(self.data)
                out = self.algorithm(data, n_clusters)
                labels = out[0]
            elif self.algorithm == kmodes.kmodes:
                out = self.algorithm(self.data, n_clusters)
                labels = out[1]
            elif self.algorithm == kmeans.kmeans:
                out = self.algorithm(self.data, n_clusters)
                labels = out[1]
            elif self.algorithm == clarans.CLARANS:
                out = self.algorithm(self.data, n_clusters)
                labels = clarans.pre_validation(out[1])
                
            DBS = davies_bouldin_score(self.data, labels)
            SHC = silhouette_score(self.data, labels)
            
            if args[0] == 'david bouldin score':
                scores.append(DBS)
            elif args[0] == 'silhouette score':
                scores.append(SHC)
            else:
                print('Please choose a valid score')

        plt.figure()
        plt.plot(range(2,crange+1), scores)
        plt.xlabel('Number of clusters')
        plt.ylabel(args[0])
        plt.title(args[1])
        plt.show()
            
    
    def library_comparison(self, labels_lib):
        adjusted_rand_score_lib = adjusted_rand_score(labels_lib, self.labels)
        normalized_mutual_info_score_lib = normalized_mutual_info_score(labels_lib, self.labels)
        print(f'Adjusted rand score library: {adjusted_rand_score_lib}')
        print(f'Normalized mutual info score library: {normalized_mutual_info_score_lib}')
        print()
        
        cmatrix = confusion_matrix(labels_lib, self.labels)
        disp_vowel = ConfusionMatrixDisplay(cmatrix)
        disp_vowel.plot()
        plt.title('Confusion matrix Library')
        plt.show()
        
    def gold_standard_comparison(self, labels_gold):
        purity = homogeneity_score(labels_gold, self.labels)
        adjusted_rand_score_gold = adjusted_rand_score(labels_gold, self.labels)
        normalized_mutual_info_score_gold = normalized_mutual_info_score(labels_gold, self.labels)
        print(f'Purity GS: {purity}')
        print(f'Adjusted rand score GS: {adjusted_rand_score_gold}')
        print(f'Normalized mutual info score GS: {normalized_mutual_info_score_gold}')
        print()
        
        cmatrix = confusion_matrix(labels_gold, self.labels)
        disp_vowel = ConfusionMatrixDisplay(cmatrix)
        disp_vowel.plot()
        plt.title('Confusion matrix Gold Standard')
        
        