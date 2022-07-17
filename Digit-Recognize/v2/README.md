#更新说明
    添加学习率优化器StepLR,实现每迭代5500次学习率减半（初始为0.003）
    （选择SGD优化器）
    accurancy稳定在99%以上（best：99.24%）

#训练（）
    python main.py -img D:/PY/hand_written/data/MNIST/raw/train-images-idx3-ubyte \
                -label D:/PY/hand_written/data/MNIST/raw/train-labels-idx1-ubyte \
                -t_img D:/PY/hand_written/data/MNIST/raw/t10k-images-idx3-ubyte \
                -t_label D:/PY/hand_written/data/MNIST/raw/t10k-labels-idx1-ubyte 
