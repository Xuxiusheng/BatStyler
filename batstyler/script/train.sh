# CUDA_VISIBLE_DEVICES=1 \
#     python batstyler/train.py \
#         --config-file batstyler/config/imagenetr.yaml \
#         --trainer KMeansPlusTrainer \
#         MODEL.BACKBONE.NAME ViT-B/32

# for i in {15..20}
# do
    CUDA_VISIBLE_DEVICES=1 \
        python batstyler/train.py \
            --config-file batstyler/config/imagenets.yaml \
            --trainer PromptLearnerTrainer \
            --output-dir batstyler/work_dirs/ETF/learner/imagenets/rn50/${i} \
            OPTIM.MAX_EPOCH 500 \
            OPTIM.LR 0.1 \
            MODEL.BACKBONE.NAME RN50 \

        
    CUDA_VISIBLE_DEVICES=1 \
        python batstyler/train.py \
            --config-file batstyler/config/imagenets.yaml \
            --trainer LinearTrainer \
            --output-dir batstyler/work_dirs/ETF/linear/imagenets/rn50/${i} \
            --root /home/xuxiusheng/deeplearning/data/ \
            OPTIM.MAX_EPOCH 80 \
            OPTIM.LR 0.002 \
            MODEL.BACKBONE.NAME RN50
# done