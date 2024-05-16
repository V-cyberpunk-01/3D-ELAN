# training for six different dataset
# if you want to train the model by yourself, you have to define the args --train to True(1)

# # dataset Nb
python main.py --data_path './0916/Nb/all_connect' \
				--train 1 \
				--com_feature_path_ego './0916/composition/composition_feature_nb_pca.csv' \
				--com_feature_path_background './0916/composition/composition_feature_nb_pca.csv' \
				--TriELAN_model_save_path  './checkpoints/nbsi-nb_' \
				--TriELAN_hidden_size 32 \
				--train_epoch 200 \
				--lr 0.000003 \
				--TriELAN_dropout_rate 0.8 \
				--wo_extra_feature True \
				--TriELAN_num_heads 8 \
				--random_seed 4 \
				--TriELAN_node_num 15 \

# # dataset nbsi-a
python main.py --data_path './0916/NbSi_a_c0_f85_k15/all_connect' \
				--train 1 \
				--com_feature_path_ego './0916/composition/composition_feature_a_pca.csv' \
				--com_feature_path_background './0916/composition/composition_feature_a_pca.csv' \
				--TriELAN_model_save_path  './checkpoints/nbsi-a_' \
				--TriELAN_hidden_size 256 \
				--train_epoch 600 \
				--lr 0.000005 \
				--wo_extra_feature False \
				--TriELAN_num_heads 12 \
				--TriELAN_hidden_size 256 \
				--random_seed 4 \
				--TriELAN_node_num 15 \
				
# dataset nbsi-b
python main.py --data_path './0916/NbSi_b_c21_f85_k17/all_connect' \
			   --train 1 \
			   --com_feature_path_ego './0916/composition/composition_feature_b_pca.csv' \
			   --com_feature_path_background './0916/composition/composition_feature_b_pca.csv' \
			   --TriELAN_model_save_path  './checkpoints/nbsi-b_' \
			   --TriELAN_hidden_size 512 \
			   --train_epoch 600 \
			   --lr 0.000005 \
			   --TriELAN_dropout_rate 0.8 \
			   --TriELAN_num_heads 12 \
			   --wo_extra_feature False \
			   --random_seed 4 \
			   --TriELAN_node_num 17 \
				
# # dataset nbsi-c
python main.py --data_path './0916/NbSi_c_c43_f85_k10/all_connect'  \
			   --train 1 \
			   --com_feature_path_ego './0916/composition/composition_feature_c_pca.csv'  \
			   --com_feature_path_background './0916/composition/composition_feature_c_pca.csv' \
			   --TriELAN_model_save_path  './checkpoints/nbsi-c_' \
			   --TriELAN_hidden_size 128  \
			   --TriELAN_num_heads 12 \
			   --train_epoch 600  \
			   --lr 0.000005  \
			   --wo_extra_feature False  \
			   --random_seed 31  \
			   --TriELAN_node_num 10 \

# # dataset nbsi-d
python main.py --data_path './0916/NbSi_d_c48_f85_k11/all_connect'  \
               --train 1 \
			   --com_feature_path_ego './0916/composition/composition_feature_d_pca.csv' \
			   --com_feature_path_background './0916/composition/composition_feature_d_pca.csv' \
			   --TriELAN_model_save_path  './checkpoints/nbsi-d_' \
			   --TriELAN_ffn_size 64 \
			   --train_epoch 500 \
			   --lr 0.000005 \
			   --wo_extra_feature False \
			   --random_seed 8 \
			   --TriELAN_node_num 11 \

# # dataset Mo
python main.py --data_path './0916/1129_Mo/shell1&2' \
			   --train 1 \
			   --com_feature_path_ego './0916/composition/composition_feature_mo_pca.csv' \
			   --com_feature_path_background './0916/composition/composition_feature_mo_pca.csv' \
			   --TriELAN_model_save_path  './checkpoints/model_mo_' \
			   --TriELAN_ffn_size 32 \
			   --train_epoch 200 \
			   --lr 0.000003 \
			   --wo_extra_feature True \
			   --random_seed 33 \
			   --TriELAN_node_num 15 \