# test for four different dataset
# if you want to train the model by yourself, you have to define the args --train to True(1)
# and add your model save path

# dataset a
python main.py --data_path './0916/NbSi_a_c0_f85_k15/all_connect' \
				--com_feature_path_ego './0916/composition/composition_feature_a_pca.csv' \
				--com_feature_path_background './0916/composition/composition_feature_a_pca.csv' \
				--train_epoch 600 \
				--lr 0.00001 \
				--random_seed 8 \
				--TriELAN_node_num 15 \
				--TriELAN_model_path 'model_basic/model_a_pca_0.956731_0.248809_0.336458_.pth' \

# dataset b
python main.py --data_path './0916/NbSi_b_c21_f85_k17/all_connect' \
				--com_feature_path_ego './0916/composition/composition_feature_b_pca.csv' \
				--com_feature_path_background './0916/composition/composition_feature_b_pca.csv' \
				--train_epoch 600 \
				--lr 0.00001 \
				--random_seed 12 \
				--TriELAN_node_num 17 \
				--TriELAN_model_path 'model_basic/model_b_rs12_pca_0.928051_0.307648_0.394738_.pth' \

# dataset c
python main.py --data_path './0916/NbSi_c_c43_f85_k10/all_connect' \
				--com_feature_path_ego './0916/composition/composition_feature_c_pca.csv' \
				--com_feature_path_background './0916/composition/composition_feature_c_pca.csv'\
				--train_epoch 600 \
				--lr 0.00001 \
				--random_seed 33 \
				--TriELAN_node_num 10 \
				--TriELAN_model_path 'model_basic/model_c_rs33_pca_0.936626_0.419159_0.584090_.pth' \

# dataset d	
python main.py --data_path './0916/NbSi_d_c48_f85_k11/all_connect' \
				--com_feature_path_ego './0916/composition/composition_feature_d_pca.csv' \
				--com_feature_path_background './0916/composition/composition_feature_d_pca.csv' \
				--train_epoch 500 \
				--lr 0.00001 \
				--random_seed 8 \
				--TriELAN_node_num 11 \
				--TriELAN_model_path 'model_basic/model_d_rs8_pca_0.903569_0.301202_0.428346_.pth' \
#