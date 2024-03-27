import os
import sys
import torch
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
plt.switch_backend('agg')
import models
from timm.models import create_model

'''
# command
python similarity_visualize.py --ckpt_path result/deit_t_LaPE/best_checkpoint.pth \
    --save_dir visualize/similarity/ --model_name deit_t_LaPE --pe learnable \
    --join-type LaPE
'''

def calculate_cosine_similarity_matrix(h_emb, eps=1e-8):
    # h_emb (N, D)
    # normalize
    a_n = h_emb.norm(dim=1).unsqueeze(1)
    a_norm = h_emb / torch.max(a_n, eps * torch.ones_like(a_n))

    # cosine similarity matrix
    sim_matrix = torch.einsum('bc,cd->bd', a_norm, a_norm.transpose(0,1))
    # sim_matrix = torch.einsum('bc,cd->bd', h_emb, h_emb.transpose(0,1))
    return sim_matrix

def get_args_parser():
    parser = argparse.ArgumentParser('Position embedding correlation viualization script', add_help=False)
    parser.add_argument('--ckpt_path', default='result/deit_t_LaPE/best_checkpoint.pth', type=str, 
                        help='The path of the model used to visualize the similarity')
    parser.add_argument('--save_dir', default='visualize/similarity/', type=str, 
                        help='The output dir')
    parser.add_argument('--model_name', default='deit_t_LaPE', type=str, 
                        choices=['deit_t_basic', 'deit_t_LaPE', 'deit_t_distill_basic', 'deit_t_distill_LaPE', 'deit_s_basic', 'deit_s_LaPE', 'deit_s_distill_basic', 'deit_s_distill_LaPE'],
                        metavar='MODEL', help='Name of model')
    parser.add_argument('--pe', default='learnable', type=str, 
                        choices=['learnable', '1D_sin', '2D_sin', '1D_RPE', '2D_RPE'],
                        metavar='NAME', help='Position embedding type in the model')
    parser.add_argument('--join-type', default='LaPE', type=str, 
                        choices=['basic', 'LaPE', 'share'],
                        metavar='NAME', help='Position embedding join type')
    return parser

def main():
    parser = argparse.ArgumentParser('Position embedding correlation viualization script', parents=[get_args_parser()])
    args = parser.parse_args()
    save_dir = os.path.join(args.save_dir,args.model_name)
    
    # model类型
    if args.model_name in ['deit_t_basic', 'deit_t_LaPE']:
        model_name = 'deit_tiny_patch16_224'
    elif args.model_name in ['deit_t_distill_basic', 'deit_t_distill_LaPE']:
        model_name = 'deit_tiny_distilled_patch16_224'
    elif args.model_name in ['deit_s_basic', 'deit_s_LaPE']:
        model_name = 'deit_small_patch16_224'
    elif args.model_name in ['deit_s_distill_basic','deit_s_distill_LaPE']:
        model_name = 'deit_small_distilled_patch16_224'
    elif args.model_name in ['deit_b_basic','deit_b_LaPE']:
        model_name = 'deit_base_patch16_224'
    elif args.model_name in ['deit_b_distill_basic','deit_b_distill_LaPE']:
        model_name = 'deit_base_distilled_patch16_224'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model = create_model(model_name,
                        pe=args.pe,
                        join_type=args.join_type,
                        pretrained=False,
                        num_classes=1000,
                        drop_rate=0.0,
                        drop_path_rate=0.1,
                        drop_block_rate=None,
                        img_size=224)
    model.load_state_dict(torch.load(args.ckpt_path, map_location='cpu')['model'])
    
    if args.pe in ['1D_RPE', '2D_RPE']:
        for i in range(model.depth):
            key1 = f'blocks.{i}.attn.relative_position_bias_table'
            key2 = f'blocks.{i}.attn.relative_position_index'
            table = model.state_dict()[key1]
            index = model.state_dict()[key2]
            relative_position_bias = table[index.view(-1)].view(14 * 14, 14 * 14, -1)
            # fig,ax = plt.subplots(1,3)
            for j in range(3):
                sns.heatmap(relative_position_bias[:,:,j])
                plt.savefig(save_dir+f'/{args.join_type}_layer{i}_head{j}.jpg')
                plt.close()

    pos_embed = torch.squeeze(model.state_dict()['pos_embed']) # (197, 768)
    pos_embed = pos_embed[1:,:] # (196, 768)
    similarity = calculate_cosine_similarity_matrix(pos_embed) # (196, 196)
    token_sim = similarity[90].reshape(14,14) # (14, 14)
    sns.heatmap(token_sim)
    plt.savefig(save_dir+'/PE_token_sim.jpg')
    plt.close()

    num_tokens, dim = pos_embed.shape # 196
    if args.join_type in ('basic', 'share'):
        dim = pos_embed.shape[-1]
        weights = torch.zeros(model.depth,dim)
        biases = torch.zeros(model.depth,dim)
        weight_ind = 0
        bias_ind = 0
        for key in model.state_dict():
            if 'norm1.weight' in key: # blocks.x.norm1.weight
                weights[weight_ind] = model.state_dict()[key]
                weight_ind += 1
            if 'norm1.bias' in key: # blocks.x.norm1.bias
                biases[bias_ind] = model.state_dict()[key]
                bias_ind += 1
        for ind in range(model.depth): 
            norm = nn.LayerNorm([dim],elementwise_affine=False)
            PE_norm = weights[ind] * norm(pos_embed) + biases[ind]
            similarity = calculate_cosine_similarity_matrix(PE_norm)
            ax = sns.heatmap(similarity)
            ax.tick_params(left=False, bottom=False)
            plt.savefig(save_dir+f'/{args.join_type}_layer{ind}_sim_overall.jpg')
            plt.close()
        # fig,ax = plt.subplots(4,3)
        for ind in range(model.depth):
            norm = nn.LayerNorm([dim],elementwise_affine=False)
            PE_norm = weights[ind] * norm(pos_embed) + biases[ind]
            similarity = calculate_cosine_similarity_matrix(PE_norm)
            token_sim = similarity[90].reshape(14,14)
            ax = sns.heatmap(token_sim, vmin=0, vmax=1.0)
            ax.tick_params(left=False, bottom=False)
            plt.savefig(save_dir + f'/{args.join_type}_layer{ind}_sim_90thToken.jpg')
            plt.close()
        # sns.heatmap(token_sim,ax=ax[ind//3,ind%3])
        # plt.savefig(save_dir+'/sim_90thToken.jpg')
        # plt.close()

    elif args.join_type=='LaPE':
        weights = torch.zeros(model.depth,dim)
        biases = torch.zeros(model.depth,dim)
        PE_norm = torch.zeros(model.depth+1,num_tokens,dim)
        PE_norm[0] = pos_embed
        # get the weights and biases of each layer's LN
        weight_ind, bias_ind = 0, 0
        for key in model.state_dict():
            if 'pe_norm.weight' in key:
                weights[weight_ind] = model.state_dict()[key]
                weight_ind += 1
            if 'pe_norm.bias' in key:
                biases[bias_ind] = model.state_dict()[key]
                bias_ind += 1
        # get the similarity of each layer's PE
        for ind in range(model.depth): 
            norm = nn.LayerNorm([dim],elementwise_affine=False)
            PE_norm[ind+1] = weights[ind] * norm(PE_norm[ind]) + biases[ind]
            similarity = calculate_cosine_similarity_matrix(PE_norm[ind+1])
            ax = sns.heatmap(similarity, vmin=0, vmax=1.0)
            ax.tick_params(left=False, bottom=False)
            plt.savefig(save_dir + f'/{args.join_type}_layer{ind}_sim_overall.jpg') # LaPE_layer0_sim.jpg
            plt.close()
        # get the similarity of the 90th token in each layer
        # fig,ax = plt.subplots(4,3)
        for ind in range(model.depth):
            norm = nn.LayerNorm([dim],elementwise_affine=False)
            PE_norm[ind+1] = weights[ind] * norm(PE_norm[ind]) + biases[ind]
            similarity = calculate_cosine_similarity_matrix(PE_norm[ind+1])
            token_sim = similarity[90].reshape(14,14)
            ax = sns.heatmap(token_sim, vmin=0, vmax=1.0)
            ax.tick_params(left=False, bottom=False)
            plt.savefig(save_dir + f'/{args.join_type}_layer{ind}_sim_90thToken.jpg')
            plt.close()
    print(f'Visualization done! Results saved in {save_dir}.')

if __name__ == '__main__':
    main()
