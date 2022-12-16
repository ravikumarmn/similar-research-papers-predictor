import json
import torch
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer
import streamlit as st
import os
import config

os.environ["TOKENIZERS_PARALLELISM"] = "false"

@st.cache
def load_model(model_str):
	sbert = SentenceTransformer(model_str,device="cpu")
	return sbert

@st.cache
def load_embeddings(embeddings_path):
	embeddings_dict = torch.load(embeddings_path)
	return embeddings_dict

@st.cache(hash_funcs={AnnoyIndex: lambda _: None})
def load_ann(ann_path,embeddings_path):
	embeddings_dict = torch.load(embeddings_path)
	embed_size = embeddings_dict["embed_size"]
	ann = AnnoyIndex(embed_size, 'angular')
	ann.load(ann_path)
	return ann
 
ann = load_ann(config.ANN_PATH,config.EMBEDDINGS_PATH)
embeddings_dict = load_embeddings(config.EMBEDDINGS_PATH)
sbert_model = load_model(config.MODEL_STR)
json_data = json.load(open(config.DATA_PATH,"r"))