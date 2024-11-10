CUDA_VISIBLE_DEVICES=1 \
        # python batstyler/train.py \
        #     --config-file batstyler/config/imagenets.yaml \
        #     --trainer KMeansPlusTrainer \
        #     --output-dir batstyler/work_dirs/ETF/kmeans/imagenets/rn50/${i} \
        #     --root /home/xuxiusheng/deeplearning/data/ \
        #     MODEL.BACKBONE.NAME RN50

        python batstyler/train.py \
            --config-file batstyler/config/imagenets.yaml \
            --trainer LLMTrainer \
            --output-dir batstyler/work_dirs/ETF/kmeans/imagenets/rn50 \
            --root /home/xuxiusheng/deeplearning/data/ \
            MODEL.BACKBONE.NAME RN50