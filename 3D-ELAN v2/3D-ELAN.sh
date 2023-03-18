# train and validation for four different dataset
# dataset a
#python main.py --data_path './1018/NbSi_a_c0_f50_k15/all_connect' \
#				--com_feature_path_ego './1018/composition/composition_feature_a_pca.csv' \
#				--com_feature_path_background './1018/composition/composition_feature_a_pca.csv' \
#				--train_epoch 600 \
#				--lr 0.00001 \
#				--random_seed 8 \
#				--TriELAN_node_num 15 \
#				--TriELAN_model_save_path './model_feature50/model_a_feature50_' \

# dataset b
#python main.py --data_path './1018/NbSi_b_c21_f50_k17/all_connect' \
#				--com_feature_path_ego './1018/composition/composition_feature_b_pca.csv' \
#				--com_feature_path_background './1018/composition/composition_feature_b_pca.csv' \
#				--train_epoch 600 \
#				--lr 0.00001 \
#				--random_seed 12 \
#				--TriELAN_node_num 17 \
#				--TriELAN_model_save_path './model_feature50/model_b_feature50_' \

# dataset c
python main.py --data_path './1018/NbSi_c_c43_f50_k10/all_connect' \
				--com_feature_path_ego './1018/composition/composition_feature_c_pca.csv' \
				--com_feature_path_background './1018/composition/composition_feature_c_pca.csv'\
				--train_epoch 600 \
				--lr 0.00001 \
				--random_seed 33 \
				--TriELAN_node_num 10 \
				--TriELAN_model_save_path './model_feature50/model_c_feature50_' \

# dataset d	
#python main.py --data_path './1018/NbSi_d_c48_f50_k11/all_connect' \
#				--com_feature_path_ego './1018/composition/composition_feature_d_pca.csv' \
#				--com_feature_path_background './1018/composition/composition_feature_d_pca.csv' \
#				--train_epoch 500 \
#				--lr 0.00001 \
#				--random_seed 8 \
#				--TriELAN_node_num 11 \
#				--TriELAN_model_save_path './model_feature50/model_d_feature50_' \
#