#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from collections import Counter
from nltk import ngrams


# In[3]:


nlp = spacy.load('en_core_web_lg') #Pretrained model to help with tokenizing


# In[4]:


PeakData=pd.read_csv('PeakData.csv')
PeakData.dropna(subset=['text'], inplace=True)


# In[5]:


PeakData.columns


# In[6]:


len(PeakData)


# In[8]:


#Remove Duplicates. Any duplication of text is dropped and one last record maintained
PeakData = PeakData.drop_duplicates(subset = ['text'], keep = 'last').reset_index(drop = True) 


# In[9]:


len(PeakData)


# In[10]:


#Function to clean text- removes digits, special chracters, and spaces
def cleanText(Text):
    Text = re.sub(r'\d+', '',str(Text)).lower()  #remove digits and convert to lower case
    Text= re.sub('[^A-Za-z0-9\s]', ' ', Text) #remove special chracters
    Text=Text.strip() #Remove leading and trailing white spaces
    return Text


# In[11]:


#messages are cleaned here
CleanedText=[]
for index,text in enumerate(PeakData['text']):
    cleanedsentence=' '.join(text.split()) #For some reason, text still had leading and trailing spaces even after cleaning. This is to explicitly remove it
    CleanedText.append(cleanText(cleanedsentence).replace("\n", "")) #Remove the next line annotation, \n
    print(index,cleanText(cleanedsentence).replace("\n", ""))


# In[12]:


PeakData['CleanedText']=CleanedText #Having cleaned messages, a new column is created for the cleaned version


# In[13]:


#Vectorize with TF-IDF-Other vectorizers can be explored
tfidf_vectorizer = TfidfVectorizer(max_df=0.85, max_features=200, stop_words='english') 
tfidf_matrix = tfidf_vectorizer.fit_transform(PeakData['text']) 
tfidf_matrix.shape


# In[14]:


# Reduce dimensionality to 2 components using PCA
from sklearn.decomposition import PCA 
pca = PCA(n_components=2)
reduced_tfidf = pca.fit_transform(tfidf_matrix.toarray()) 
reduced_tfidf.shape


# In[16]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
#Using the Elbow method to detrmine optimal number of clusters
wcss = []  # within-cluster sum of squares
cluster_range = range(1, 10)  # test up to 10 clusters 
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(reduced_tfidf)
    wcss.append(kmeans.inertia_)
 
#Plot to detrmine elbow
plt.figure(figsize=(10, 6))
plt.plot(cluster_range, wcss, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.grid(True)
plt.show()


# In[17]:


# Perform KMeans clustering with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(reduced_tfidf)
 
plt.figure(figsize=(10, 6))
plt.scatter(reduced_tfidf[:, 0], reduced_tfidf[:, 1], c=clusters, cmap='rainbow')
#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1])
            #s=200, c='black', marker='o', label='Centroids')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Volume of Telgram messages about Anti-Vaccination')
plt.legend()
plt.grid(True)
plt.show()


# In[18]:


PeakData['Cluster']=clusters #Add a column of corresponding cluster


# In[35]:


#Determine the top 10 n-word phrases from each cluster. For each of the clusters, form one string that will be used to determined the top n-word phrases
Merged=pd.DataFrame()
for cluster in PeakData['Cluster'].unique():
    OneString=''
    currentdf=PeakData[PeakData['Cluster']==cluster]['CleanedText']
    for text in currentdf:   
        text=nlp(text)
        for token in text:            
            if token.is_stop==False:
                OneString=OneString+' '+token.lemma_ #For One strin from each cluster
    phrases=[]
    for token in ngrams(OneString.split(), 3): #Get 3 word phrases- this can be changed         
        phrases.append(' '.join(token))
    TopPhrases=Counter(phrases).most_common(10) #Obtain the top 10
    ClusterTopwords=pd.DataFrame(TopPhrases,columns=('Top phrases','Volume'))
    ClusterTopwords['Cluster'] = pd.Series([cluster for x in range(len(ClusterTopwords.index))]) 
    Merged=pd.concat([Merged,ClusterTopwords])
            
    print(cluster,len(currentdf))
        


# In[33]:


Merged


# In[34]:


Merged.to_csv('3word.csv',index=False)


# In[37]:


PeakData.to_csv('Overall.csv',index=False)


# In[ ]:




