from urllib.request import urlopen
import re
import os
import requests
from transformers import BertModel, BertTokenizer
from sklearn.cluster import KMeans
import torch
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle

def find_json_urls(html):
    # It finds all the coincidences to the pattern "url":"<url>"
    pattern = r'"url":"([^"]+)"'
    urls = re.findall(pattern, html)
    return urls

def get_xml_source(url):
    """
    This function get the url sources to download the data in xml format
    """
    page = urlopen(url)
    html = page.read().decode("utf-8")

    all_urls = find_json_urls(html)
    xls_urls = []

    for link in all_urls:
        if link.endswith(".xls"):
            xls_urls.append(link)

    return xls_urls

def download_data(xls_urls, target_folder):
    xls_paths = []
    for i, url in enumerate(xls_urls):
        try:
            # get the file name from the url
            file_name = os.path.basename(url)

            target_path = os.path.join(target_folder, file_name)

            if os.path.isfile(target_path):
                print(f"[{100*(i+1)/len(xls_urls):.2f}%] XLS file {file_name} already downloaded")
                xls_paths.append(target_path)
                continue

            # get the content in the file
            xls_file = requests.get(url)

            # save the xls file in the target folder
            with open(target_path, 'wb') as f:
                f.write(xls_file.content)

            print(f"[{100*(i+1)/len(xls_urls):.2f}%] XLS file {file_name} correctly downloaded")
            xls_paths.append(target_path)

        except Exception as e:
            print(f"Error while trying to download the file {file_name}: {str(e)}")

    return xls_paths

def texts_to_embeddings(strings):
	# Inicializar BERT tokenizer y modelo
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	model = BertModel.from_pretrained('bert-base-uncased')
	
	# Generar embeddings con BERT
	embeddings = []
	for string in strings:
	    inputs = tokenizer(string, return_tensors='pt', padding=True, truncation=True)
	    outputs = model(**inputs)
	    embeddings.append(outputs.last_hidden_state.mean(1).squeeze().detach().numpy())

	return embeddings

def kmeans_elbow(X, max_clusters=40):
    distortions = []
    K = range(1,max_clusters+1)
    for k in K:
        kmeanModel = KMeans(n_clusters=k, random_state=42)
        kmeanModel.fit(X)
        distortions.append(kmeanModel.inertia_)
    
    plt.figure(figsize=(16,8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Inertia of the KMeans Algorithm')
    plt.show()

def clustering_texts(embeddings, num_clusters):
	
	# Aplicar K-means a los embeddings
	km = KMeans(n_clusters=num_clusters, random_state=42)
	km.fit(embeddings)
	
	# Asignar las etiquetas de cluster a tus strings
	clusters = km.labels_
	
	return clusters

def save_to_pickle(obj, path):
    with open(path, 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

def read_pickle(path):
    with open(path, 'rb') as file:
        obj = pickle.load(file)
    return obj