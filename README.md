# Layer-adaptive-Position-Embedding
## Introdcution
LaPE is a kind of new Position Embedding joining method, and it is compatible to most of the Vision Transformer. Here we use the codebase of [DeiT](https://github.com/facebookresearch/deit) to illustrate how to adding LaPE to Vision Transformer.

## Train the model
```
python -m torch.distributed.launch --nproc_per_node=[num_gpus] --master_port [port] --use_env main.py --model [model type] --batch-size [bs] --data-path [path to dataset] --output_dir [path to output] --adding-type [PE joining method] 
```
for example
```
python -m torch.distributed.launch --nproc_per_node=4 --master_port 29400 --use_env main.py --model deit_tiny_patch16_224 --batch-size 256 --data-path data/imagenet --output_dir result/deit_t_LaPE/ --adding-type LaPE
```
## Test the model
```
python main.py --eval --resume [path to model] --model deit_small_patch16_224 --data-path [path to dataset] --adding-type [PE joining method] 
```
for example
```
python main.py --eval --resume result/deit_t_LaPE/best_checkpoint.pth --model deit_tiny_patch16_224 --data-path data/imagenet --adding-type LaPE
```
