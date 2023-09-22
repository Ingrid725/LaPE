import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
plt.switch_backend('agg')
import models
from timm.models import create_model


def calculate_cosine_similarity_matrix(h_emb, eps=1e-8):
    # h_emb (N, M)
    # normalize
    a_n = h_emb.norm(dim=1).unsqueeze(1)
    a_norm = h_emb / torch.max(a_n, eps * torch.ones_like(a_n))

    # cosine similarity matrix
    sim_matrix = torch.einsum('bc,cd->bd', a_norm, a_norm.transpose(0,1))
    return sim_matrix

def main():
    # checkpoint path
    ckpt = './result/deit_t_default_2dRPE/best_checkpoint.pth'
    save_dir = './visualize/'
    
    file_name = ckpt.split('/')[-2]
    save_dir = os.path.join(save_dir,file_name)

    # pe
    pe = '1D_sin' # ['learnable','1D_sin','2D_sin','RPE']

    # pe joining method
    pe_joining = 'default' # ['default', 'LaPE']
    
    depth = 12

    # model类型
    if 'deit_t' in file_name:
        model_name = 'deit_tiny_patch16_224'
    elif file_name in ('deit_t_distill_default','deit_t_distill_LaPE'):
        model_name = 'deit_tiny_distilled_patch16_224'
    elif file_name in ('deit_s_default','deit_s_LaPE'):
        model_name = 'deit_small_patch16_224'
    elif file_name in ('deit_s_distill_default','deit_s_distill_LaPE'):
        model_name = 'deit_small_distilled_patch16_224'
    elif file_name in ('deit_b_default','deit_b_LaPE'):
        model_name = 'deit_base_patch16_224'
    elif file_name in ('deit_b_distill_default','deit_b_distill_LaPE'):
        model_name = 'deit_base_distilled_patch16_224'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model = create_model(model_name,
                        pe=pe,
                        pe_joining=pe_joining,
                        pretrained=False,
                        num_classes=1000,
                        drop_rate=0.0,
                        drop_path_rate=0.1,
                        drop_block_rate=None,
                        img_size=224)
    model.load_state_dict(torch.load(ckpt, map_location='cpu')['model'])
    pos_embed = torch.squeeze(model.state_dict()['pos_embed'])
    pos_embed = pos_embed[1:,:]

    # visualize for VTs with default PE
    if pe_joining=='default':
        similarity = calculate_cosine_similarity_matrix(pos_embed)
        sns.heatmap(similarity)
        plt.savefig(save_dir+'/initial_pe.jpg' %pe_joining)
        plt.close()

        dim = pos_embed.shape[-1]
        weights = torch.zeros(depth,dim)
        biases = torch.zeros(depth,dim)
        weight_ind = bias_ind = 0
        for key in model.state_dict():
            # blocks.x.norm1.weight
            if 'norm1.weight' in key:
                weights[weight_ind] = model.state_dict()[key]
                weight_ind += 1
            # blocks.x.norm1.bias
            if 'norm1.bias' in key:
                biases[bias_ind] = model.state_dict()[key]
                bias_ind += 1
        for ind in range(depth): 
            norm = nn.LayerNorm([dim],elementwise_affine=False)
            PE_norm = weights[ind] * norm(pos_embed) + biases[ind]
            similarity = calculate_cosine_similarity_matrix(PE_norm)
            sns.heatmap(similarity, vmin=0, vmax=1.0)
            plt.savefig(save_dir+'/default_sim_layer%s.jpg' %ind)
            plt.close()
            token_sim = similarity[90].reshape(14,14)
            sns.heatmap(token_sim, vmin=0, vmax=1.0)
            plt.savefig(save_dir+'/default_token_sim_layer%s.jpg' %ind)
            plt.close()

    # visualize for VTs with LaPE
    elif pe_joining=='LaPE':
        dim = pos_embed.shape[-1]
        weights = torch.zeros(12,dim)
        biases = torch.zeros(12,dim)
        weight_ind = 0
        bias_ind = 0
        for key in model.state_dict():
            # blocks.x.pe_norm.weight
            if 'pe_norm.weight' in key:
                weights[weight_ind] = model.state_dict()[key]
                weight_ind += 1
            # blocks.x.pe_norm.bias'
            if 'pe_norm.bias' in key:
                biases[bias_ind] = model.state_dict()[key]
                bias_ind += 1
        for ind in range(12):
            norm = nn.LayerNorm([dim],elementwise_affine=False)
            PE_norm = weights[ind] * norm(pos_embed) + biases[ind]
            similarity = calculate_cosine_similarity_matrix(PE_norm)
            sns.heatmap(similarity,vmin=0,vmax=1.0)
            plt.savefig(save_dir+'/LaPE_sim_layer%s.jpg' %ind)
            plt.close()
            token_sim = similarity[90].reshape(14,14)
            sns.heatmap(token_sim,vmin=0,vmax=1.0)
            plt.savefig(save_dir+'/LaPE_token_sim_layer%s.jpg' %ind)
            plt.close()
    
    # visualize for VTs with RPE
    # elif pe=='RPE':
    #     for i in range(depth):
    #         key1 = f'blocks.{i}.attn.relative_position_bias_table'
    #         key2 = f'blocks.{i}.attn.relative_position_index'
    #         table = model.state_dict()[key1]
    #         index = model.state_dict()[key2]
    #         relative_position_bias = table[index.view(-1)].view(14 * 14, 14 * 14, -1)
    #         for j in range(3):
    #             sns.heatmap(relative_position_bias[:,:,j])
    #             plt.savefig(save_dir+f'/RPE_layer{i}_head{j}.jpg')
    #             plt.close()



if __name__ == '__main__':
    main()
