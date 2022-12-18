import streamlit as st
from model import sbert_model, ann, json_data, embeddings_dict

st.title("Search NeurIPS-2022 Papers")

question = st.text_input("Enter a paper identifier",placeholder='For example : "Attention Is All You Need"')
topk = st.slider("TopK",1,10,5)
if question:
	sbert_embeds = sbert_model.encode(question,show_progress_bar=False,device="cpu")
	results = ann.get_nns_by_vector(sbert_embeds, topk, search_k=-1, include_distances=False)
	for res in results:
		id = embeddings_dict["model_id_to_paper_id"][res]
		out = json_data[str(id)]
		st.write(
			"["+out["title"]+"]"+\
				"("+out["url"]+")"
			)
	question = None

