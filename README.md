Code inplement of dual-graph sequential model and finetuning LLaMA2 of URLLM

./DG_Final/GM processed movie-game dataset

./DG_Final/AO processed art-office dataset

./DG_Final path of Dual Graph Sequence Modeling Model of URLLM

./DG_Final/DG_src/dataset/Entertainment-Education_Amazon  user interaction in movie-game dataset

./DG_Final/DG_src/dataset/Entertainment-Education_AO  user interaction in art-office dataset

./DG_Final/DG_src/dataset/item_prompt_GM item attribute in movie-game dataset(generated from GPT)

./DG_Final/DG_src/dataset/item_prompt_AO item attribute in art-official dataset(generated from GPT)

./llama2-SFT path of finetuning LLaMA2

the following are pipeline of running URLLM on movie-game dataset. The same as art-office dataset.

python ./DG_Final/GM/jsonBuilder_attribute_graph_GPT.py to gain prompts for GPT to build attribute graph

python ./DG_Final/GM/testGPT35 to gain prompts for GPT to build attribute graph

run sh ./DG_Final/DG_src/train.sh to gain the similarity of movie-game dataset user to generate ./DG_Final/DG_src/t4_G2_final_DGresult_matmul_trte.npy

python ./DG_Final/GM/jsinBuilder_testing_rc.py utilizing similar user to gain prompt for LLaMA2
finally sh ./llama2-SFT/generate.sh to gain answers from LLM.
