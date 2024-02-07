Code inplement of dual-graph sequential model and finetuning LLaMA2 of URLLM

./DG_Final/GM processed movie-game dataset
./DG_Final/AO processed art-office dataset

the following are pipeline of running URLLM on movie-game dataset. The same as art-office dataset.

python ./DG_Final/GM/jsonBuilder_attribute_graph_GPT.py to gain prompts for GPT to build attribute graph
python ./DG_Final/GM/testGPT35 to gain prompts for GPT to build attribute graph
./DG_Final Dual Graph Sequence Modeling Model of URLLM
run sh ./DG_Final/DG_src/train.sh to gain the similarity of movie-game dataset user in generated ./DG_Final/DG_src/t4_G2_final_DGresult_matmul_trte.npy

python ./DG_Final/GM/jsinBuilder_testing_rc.py to gain prompt for LLaMA2
finally sh ./LLaMA_SFT/Generate.sh to gain answers from LLM.
