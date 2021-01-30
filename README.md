# transformers-gpt2-Chinese-New-Year-Couplets
基于gpt2模型生成春节对联
## 安装支持库
pip3 install -r requirements.txt
## 训练步骤
1.python3 new_data.py

2.python3 train.py

## 注意：
若是训练时loss下降缓慢或者很难下降，将loss直接改为keras.losses.SparseCategoricalCrossentropy()，或者将optimizer改为默认的adam

# transformers库的链接

https://huggingface.co/transformers/

# 测试图在image文件中。训练了100轮，loss最终下降到了0.2左右。
