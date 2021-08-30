#!/usr/bin/env bash

datasets=("Fdataset" "Cdataset" "lrssl" "lagcn")
comment="test2"
epochs=800
neighbor_num=15
lr=1e-2
embedding_dim=128
for dataset in ${datasets[*]};do
      python demo.py --dataset_name ${dataset} --neighbor_num ${neighbor_num} --comment ${comment} \
      --epochs ${epochs} --drug_neighbor_num ${neighbor_num} --disease_neighbor_num ${neighbor_num} \
      --embedding_dim ${embedding_dim} --lr ${lr}
done

comment="test"
epochs=400
neighbor_num=15
lr=5e-3
embedding_dim=128
for dataset in ${datasets[*]};do
      python demo.py --dataset_name ${dataset} --neighbor_num ${neighbor_num} --comment ${comment} \
      --epochs ${epochs} --drug_neighbor_num ${neighbor_num} --disease_neighbor_num ${neighbor_num} \
      --embedding_dim ${embedding_dim} --lr ${lr}
done