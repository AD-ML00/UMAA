# UMAA
UMAA anomaly detection approach

Unsupervised Multi-head Attention Autoencoder for Multivariate Time-Series Anomaly Detection

requirements
- python 3.8.17
- torch 2.0.1+cu117

usage
$python main.py -mn UMAA -d SMAP -w 12 -e 10 -b 32 
#mn : model name, d : dataset, w : window size, e : epoch, b : batch size
