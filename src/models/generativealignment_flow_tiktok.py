# coding: utf-8
# rongqing001@e.ntu.edu.sg
r"""
SMORE - Multi-modal Recommender System
Reference:
    ACM WSDM 2025: Spectrum-based Modality Representation Fusion Graph Convolutional Network for Multimodal Recommendation

Reference Code:
    https://github.com/kennethorq/SMORE
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph
from torch.nn.utils import spectral_norm


# 这个是生成函数的采样函数（不用改）
class LogitNormalSampler:
    def __init__(self, normal_mean: float = 0, normal_std: float = 1):
        # follows https://arxiv.org/pdf/2403.03206.pdf
        # sample from a normal distribution
        # pass the output through standard logistic function, i.e., sigmoid
        self.normal_mean = float(normal_mean)
        self.normal_std = float(normal_std)

    @torch.no_grad()
    def sample_time(self, x_start):
        x_normal = torch.normal(
            mean=self.normal_mean,
            std=self.normal_std,
            size=(x_start.shape[0],),
            device=x_start.device,
        )
        x_logistic = torch.nn.functional.sigmoid(x_normal)
        return x_logistic

# 这个是模态生成函数类（暂时不用改）
class FlowModel(nn.Module):
    def __init__(
        self,
        input_feat_dim,
        dim_feat,
        time_emb_size: int,
        norm=False,
        init_dropout=0,
        sigma_min =  1e-5,
        sigma_max = 1.0
    ):
        super(FlowModel, self).__init__()

        self.time_emb_dim = time_emb_size
        self.norm = norm
        self.input_feat_dim = input_feat_dim
        self.dim_feat = dim_feat
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        self.sample_timescale = self.sigma_max  - self.sigma_min

        self.time_emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        self.num_time_steps = 20
        self.alpha = 20

        self.time_step_sampler = LogitNormalSampler()

        self.mlp_layers = nn.Sequential(
            nn.Linear(2 * self.dim_feat + self.time_emb_dim, self.dim_feat),
            nn.Tanh(),
            nn.Dropout(0.1),
            
            nn.Linear(self.dim_feat, self.dim_feat),
            nn.Tanh(),
            nn.Dropout(0.1),

            nn.Linear(self.dim_feat, self.dim_feat)
        )

        self.init_dropout = nn.Dropout(init_dropout)

        self.x_proj = nn.Linear(self.dim_feat + self.time_emb_dim, self.dim_feat)

        # condition context
        # self.time_emb_layer = nn.Linear(time_emb_size, dim_feat)
        self.cond_proj = nn.Linear(dim_feat, dim_feat)

        # lightweight modulation
        self.scale_proj = nn.Linear(dim_feat, dim_feat)
        self.shift_proj = nn.Linear(dim_feat, dim_feat)
        self.gate_proj = nn.Linear(dim_feat, dim_feat)

        # 2-layer residual MLP
        self.fc1 = nn.Linear(dim_feat, dim_feat)
        self.fc2 = nn.Linear(dim_feat, dim_feat)
        self.out = nn.Linear(dim_feat, dim_feat)

        self.dropout = nn.Dropout(0.1)
        self.norm1 = nn.LayerNorm(dim_feat)
        self.norm2 = nn.LayerNorm(dim_feat)

        

    def modulate(self, h, ctx):
        scale = torch.tanh(self.scale_proj(ctx))
        shift = self.shift_proj(ctx)
        gate = torch.sigmoid(self.gate_proj(ctx))
        return h + gate * (h * scale + shift)



    def forward(self, x, cond, t):

        time_emb = timestep_embedding_pi(t, self.time_emb_dim).to(x.device)

        emb = self.time_emb_layer(time_emb)

        h = torch.cat([x, emb], dim=-1)

        h = self.x_proj(h)

        res = h
        h = self.modulate(h, cond)
        h = self.norm1(h)
        h = self.dropout(h)
        out = h + res

        # time_emb = timestep_embedding_pi(t, self.time_emb_dim).to(x.device)

        # emb = self.time_emb_layer(time_emb)     
        # if self.norm:
        #     x = F.normalize(x)
        # x = self.init_dropout(x)
        # h = torch.cat([x, user_condition, emb], dim=-1)
        # output = self.mlp_layers(h)
        return out

    def drop_condition(self, cond, p=0.2, training=True):
        if (not training) or p <= 0:
            return cond
        mask = (torch.rand(cond.size(0), 1, device=cond.device) > p).float()
        return cond * mask
    
    def compute_loss(self, start, target, user_condition=None):
        
        # train id2t flow model
        sample_time = self.time_step_sampler.sample_time(start)

        xt = self.psi(sample_time, x=start, x1=target)
        
        target_velocity = self.Dt_psi(x=start, x1=target)

        self.guidance_scale = 0.5

        cond_in = self.drop_condition(user_condition, training=self.training)
        condition_prediction = self.forward(xt, cond_in, sample_time.squeeze(-1))

        null_condition = torch.zeros_like(user_condition)
        uncondition_prediction = self.forward(xt, null_condition, sample_time.squeeze(-1))

        prediction = uncondition_prediction + self.guidance_scale * (condition_prediction - uncondition_prediction)


        # log_snr = 4 - t * 8 # compute from timestep : inversed
        # condition_prediction = self.forward(xt, user_condition, sample_time.squeeze(-1))

        # null_user_condition = torch.zeros_like(user_condition)

        # uncondition_prediction = self.forward(xt, null_user_condition, sample_time.squeeze(-1))

        # prediction = ( 1 + 0.2 ) * condition_prediction - 0.2 * uncondition_prediction

        alignment_loss = self.mos(target_velocity - prediction).mean()

        Xt = start + prediction * sample_time.unsqueeze(-1).repeat(1, prediction.shape[1])

        return Xt, alignment_loss
    

    def inference(self, start, user_condition):
        
        # start textual feature 
        Xt = start

        # 将时间步的区间划分为num_time_steps个步骤
        sigma_steps = torch.linspace( self.sigma_min, self.sigma_max, self.num_time_steps, device=start.device )

        # 用户偏好的时间步数
        discrete_time_steps_to_eval_model_at = torch.linspace(
            self.sigma_min, self.sigma_max, self.num_time_steps, device=Xt.device
        )

        time_velocity = []
        for i in range(self.num_time_steps - 1):
            
            # 这个的i指的是相对于 self.num_time_steps的索引，还需要映射回实际的时间步
            i_t = discrete_time_steps_to_eval_model_at[i]

            velocity = self.forward(Xt, user_condition, i_t.repeat(Xt.shape[0]) )

            step_size = sigma_steps[i + 1] - sigma_steps[i]

            # print("velocity:", Xt.shape, velocity.shape, step_size.shape, W[:,i].shape, flush=True)

            Xt = Xt + velocity * step_size

            time_velocity.append(velocity * step_size)

        time_velocity = torch.stack(time_velocity, dim=0)

        return Xt, time_velocity
        

    ## flow matching specific functions
    def psi(self, t, x, x1):
        assert (
            t.shape[0] == x.shape[0]
        ), f"Batch size of t and x does not agree {t.shape[0]} vs. {x.shape[0]}"
        assert (
            t.shape[0] == x1.shape[0]
        ), f"Batch size of t and x1 does not agree {t.shape[0]} vs. {x1.shape[0]}"

        # print(x.shape, x1.shape, t, t.shape, t.ndim, flush=True)
        assert t.ndim == 1
        
        t = self.expand_t(t, x)

        return (t * (self.sigma_min / self.sigma_max - 1) + 1) * x + t * x1

    def Dt_psi(self, x: torch.Tensor, x1: torch.Tensor):
        assert x.shape[0] == x1.shape[0]
        return (self.sigma_min / self.sigma_max - 1) * x + x1

    def expand_t(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t_expanded = t
        while t_expanded.ndim < x.ndim:
            t_expanded = t_expanded.unsqueeze(-1)
        return t_expanded.expand_as(x)

    def mos(self, err, start_dim=1, con_mask=None):  # mean of square
        if con_mask is not None:
            return (err.pow(2).mean(dim=-1) * con_mask).sum(dim=-1) / con_mask.sum(dim=-1)
        else:
            return err.pow(2).flatten(start_dim=start_dim).mean(dim=-1)


def timestep_embedding_pi(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(
        timesteps.device
    ) * 2 * math.pi  # shape (dim//2,)
    args = timesteps[:, None].float() * freqs[None]  # (N, dim//2)
    embedding = torch.cat(
        [torch.cos(args), torch.sin(args)], dim=-1)  # (N, (dim//2)*2)
    if dim % 2:
        # zero pad in the last dimension to ensure shape (N, dim)
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding




# 这个是modality的encoder，用来将modality feature 映射为均值和方差
class ModalityVariationalEncoder(nn.Module):
    """有效性保证的VAE"""
    def __init__(self, input_dim=512, latent_dim=256):
        super().__init__()
        
        # === 有效性保证1：Spectral Normalization ===

        self.fc1 = spectral_norm(nn.Linear(input_dim, 4 * latent_dim))
        self.fc2 = spectral_norm(nn.Linear(4 * latent_dim, 4 * latent_dim))
        

        self.fc_mu = nn.Linear(4 * latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(4 * latent_dim, latent_dim)
        
        # Decoder（用于重建）
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 4 * latent_dim),
            nn.ReLU(),
            nn.Linear(4 * latent_dim, 4 * latent_dim),
            nn.ReLU(),
            nn.Linear(4 * latent_dim, input_dim)
        )
    
    def get_linear_layers(self, x):
        with torch.no_grad():
            h = F.relu(self.fc1(x))
            h = F.relu(self.fc2(h))
        return h

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def compute_loss(self, x, z, mu, logvar, kl_weight=1e-2, free_bits=0.5):
        """计算VAE loss，包含多重保护"""
        
        # === 有效性保证3：Reconstruction Loss ===
        recon = self.decoder(z)
        L_recon = F.mse_loss(recon, x)
        
        # === 有效性保证4：Free Bits KL ===
        # L_kl = -0.5 * torch.sum(1 + logvar - (0.3 * mu) ** 6 - logvar.exp(), dim = 1).mean()
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
        L_kl = kl_per_dim.sum(dim=-1).mean()
        
        # === 有效性保证5：KL Annealing ===
        L_total = L_recon + kl_weight * L_kl
        
        return L_total, L_recon, L_kl
    
# 这个是推荐的主要类
class GenerativeAlignment(GeneralRecommender):
    def __init__(self, config, dataset):
        super(GenerativeAlignment, self).__init__(config, dataset)
        self.sparse = True
        self.cl_loss = config['cl_loss']
        self.n_ui_layers = config['n_ui_layers']
        self.embedding_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.image_knn_k = config['image_knn_k']
        self.text_knn_k = config['text_knn_k']
        self.audio_knn_k = config['audio_knn_k']
        self.dropout_rate = config['dropout_rate']
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.mask_weight_g = config['mask_weight_g']
        self.mask_weight_f = config['mask_weight_f']
        self.test_missing_rate = config['missing_rate']
        self.alpha = config['edge_alpha']


        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.epoch_counter = 0

        # 初始化行为的embedding
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        

        # 行为矩阵的正则化
        self.norm_adj = self.get_adj_mat()
        self.R_sprse_mat = self.R
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)
        
        # 采样确定哪些item是缺失模态的
        self.preprocess_test_missing_modal(missing_rate=self.test_missing_rate)


        # 模态feature 加载，实例化生成函数
        if self.v_feat is not None:
            self.vae_v_model = ModalityVariationalEncoder(self.v_feat.shape[1], self.embedding_dim)
            self.z2v_flow_model = FlowModel(input_feat_dim=self.embedding_dim, dim_feat=self.embedding_dim, time_emb_size= 10)

        if self.t_feat is not None:
            self.vae_t_model = ModalityVariationalEncoder(self.t_feat.shape[1], self.embedding_dim)
            self.z2t_flow_model = FlowModel(input_feat_dim=self.embedding_dim, dim_feat=self.embedding_dim, time_emb_size= 10)

        if self.a_feat is not None:
            self.vae_a_model = ModalityVariationalEncoder(self.a_feat.shape[1], self.embedding_dim)
            self.z2a_flow_model = FlowModel(input_feat_dim=self.embedding_dim, dim_feat=self.embedding_dim, time_emb_size= 10)

       
        # 初始化用来构建item-item图的代码
        if self.v_feat is not None:

            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.trans_image_embedding = nn.Parameter(nn.init.xavier_normal_(torch.tensor(np.random.randn(self.v_feat.shape[0], self.embedding_dim), dtype=torch.float32, requires_grad=True)))

            image_adj = build_sim(self.image_embedding.weight.detach())
            image_adj = build_knn_normalized_graph(image_adj, topk=self.image_knn_k, is_sparse=self.sparse,
                                                       norm_type='sym')
            v_missing_idx = (self.test_visual_missing_mask == 0).cpu().numpy()
            image_adj = image_adj.to_dense()
            image_adj[v_missing_idx, :] = 0.0
            image_adj[:, v_missing_idx] = 0.0
            image_adj[v_missing_idx, v_missing_idx] = 1.0
            self.image_original_adj = image_adj.to_sparse_coo().cuda()

        if self.t_feat is not None:

            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.trans_text_embedding = nn.Parameter(nn.init.xavier_normal_(torch.tensor(np.random.randn(self.t_feat.shape[0], self.embedding_dim), dtype=torch.float32, requires_grad=True)))

            text_adj = build_sim(self.text_embedding.weight.detach())
            text_adj = build_knn_normalized_graph(text_adj, topk=self.text_knn_k, is_sparse=self.sparse, norm_type='sym')

            t_missing_idx = (self.test_textual_missing_mask  == 0).cpu().numpy()
            text_adj = text_adj.to_dense()
            text_adj[t_missing_idx, :] = 0.0
            text_adj[:, t_missing_idx] = 0.0
            text_adj[t_missing_idx, t_missing_idx] = 1.0
            self.text_original_adj = text_adj.to_sparse_coo().cuda()

        if self.a_feat is not None:

            self.audio_embedding = nn.Embedding.from_pretrained(self.a_feat, freeze=False)
            self.trans_audio_embedding = nn.Parameter(nn.init.xavier_normal_(torch.tensor(np.random.randn(self.a_feat.shape[0], self.embedding_dim), dtype=torch.float32, requires_grad=True)))

            audio_adj = build_sim(self.audio_embedding.weight.detach())
            audio_adj = build_knn_normalized_graph(audio_adj, topk=self.audio_knn_k, is_sparse=self.sparse, norm_type='sym')

            a_missing_idx = (self.test_audio_missing_mask == 0).cpu().numpy()
            audio_adj = audio_adj.to_dense()
            audio_adj[a_missing_idx, :] = 0.0
            audio_adj[:, a_missing_idx] = 0.0
            audio_adj[a_missing_idx, a_missing_idx] = 1.0
            self.audio_original_adj = audio_adj.to_sparse_coo().cuda()


        # 一些MLP的映射层
        self.query_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        )
        self.query_t = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        )

        self.gate_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_t = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_a = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.prior_mu = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.prior_logvar = nn.Parameter(torch.zeros(1, self.embedding_dim))

    # 传统行为矩阵的
    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        return norm_adj_mat.tocsr()

    # 矩阵格式转换的
    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    

    def preprocess_test_missing_modal(self, missing_rate):
        
        n_missing = int(self.n_items * missing_rate)
        self.test_miss_items = []

        visual_mask = np.ones(self.n_items, dtype=np.int32)
        textual_mask = np.ones(self.n_items, dtype=np.int32)
        audio_mask = np.ones(self.n_items, dtype=np.int32)

        missing_type = np.random.randint(0, 4, size=n_missing)
        missing_items = np.random.choice(self.n_items, n_missing, replace=False)

        visual_mask[missing_items[missing_type == 0]] = 0
        textual_mask[missing_items[missing_type == 1]] = 0
        audio_mask[missing_items[missing_type == 2]] = 0


        visual_mask[missing_items[missing_type == 3]] = 0
        textual_mask[missing_items[missing_type == 3]] = 0
        audio_mask[missing_items[missing_type == 3]] = 0


        self.test_miss_items = missing_items

        # missing_items = np.random.choice(self.n_items, n_missing, replace=False)
        # visual_mask[missing_items] = 0
        # self.test_miss_items.append(missing_items)
        # missing_items = np.random.choice(self.n_items, n_missing, replace=False)
        # textual_mask[missing_items] = 0
        # self.test_miss_items.append(missing_items)
        # missing_items = np.random.choice(self.n_items, n_missing, replace=False)
        # visual_mask[missing_items] = 0
        # textual_mask[missing_items] = 0
        # self.test_miss_items.append(missing_items)
        
        # self.test_miss_items = np.unique(np.concatenate(self.test_miss_items))

        visual_observed_idx = (visual_mask == 1)
        text_observed_idx = (textual_mask == 1)
        audio_observed_idx = (audio_mask == 1)

        visual_missing_idx = (visual_mask == 0)
        textual_missing_idx = (textual_mask == 0)
        audio_missing_idx = (audio_mask == 0)

        image_mean = self.v_feat[visual_observed_idx].mean(dim=0)
        text_mean = self.t_feat[text_observed_idx].mean(dim=0)
        audio_mean = self.a_feat[audio_observed_idx].mean(dim=0)
        
        self.v_feat[visual_missing_idx] = image_mean
        self.t_feat[textual_missing_idx] = text_mean
        self.a_feat[audio_missing_idx] = audio_mean
    

        self.test_visual_missing_mask = torch.tensor(visual_mask).to(self.device).int()
        self.test_textual_missing_mask = torch.tensor(textual_mask).to(self.device).int()
        self.test_audio_missing_mask = torch.tensor(audio_mask).to(self.device).int()
        
        print("Test - V missing: {}, T missing: {}, A missing: {}".format(
                (visual_mask == 0).sum(), (textual_mask == 0).sum(), (audio_mask == 0).sum()))
        

    # 用来更新模态item-item邻接矩阵的
    def update_adj(self, train=True):
        with torch.no_grad() :
            # 1. 获得模态缺失的物品 index

            torch.cuda.empty_cache()
            self.image_original_adj = self.image_original_adj.cpu().to_dense()

            # 2. 根据当前模态表征信息重新计算item-item之间的关系
            image_adj = build_sim(self.trans_image_embedding.detach())
            image_adj = build_knn_neighbourhood(image_adj, topk=self.image_knn_k)
            image_adj = compute_normalized_laplacian(image_adj).cpu()
            
            self.image_original_adj = (1- self.alpha) * self.image_original_adj + self.alpha * image_adj
             # 3. 只更新模态缺失的那些item
            # if train:
            #     # pass
            #     # 为了训练的稳定性，会用完整模态的边信息作为基础
            #     # print(self.image_original_adj[v_missing_mask, :][0])
            #     self.image_original_adj = (1- self.alpha) * self.image_original_adj + self.alpha * image_adj
            # else:
            #     # 测试的时候会直接用根据补齐模态重构的边信息
            #     self.image_original_adj = image_adj
                # pass
           
            self.image_original_adj = self.image_original_adj.to_sparse_coo()
            del image_adj


        with torch.no_grad() :
            # 开始处理文本模态
            torch.cuda.empty_cache()
            self.text_original_adj = self.text_original_adj.cpu().to_dense()


            text_adj = build_sim(self.trans_text_embedding.detach())
            text_adj = build_knn_neighbourhood(text_adj, topk=self.text_knn_k)
            text_adj = compute_normalized_laplacian(text_adj).cpu()

            self.text_original_adj = (1- self.alpha) * self.text_original_adj + self.alpha * text_adj
            # # 3. 只更新模态缺失的那些item
            # if train:
            #     # 为了训练的稳定性，会用完整模态的边信息作为基础
            #     self.text_original_adj = (1- self.alpha) * self.text_original_adj + self.alpha * text_adj

            # else:
            #     # 测试的时候会直接用根据补齐模态重构的边信息
            #     self.text_original_adj = text_adj

            self.text_original_adj = self.text_original_adj.to_sparse_coo()
            del text_adj

        with torch.no_grad() :
            # 开始处理文本模态
            torch.cuda.empty_cache()
            self.audio_original_adj = self.text_original_adj.cpu().to_dense()

            audio_adj = build_sim(self.trans_audio_embedding.detach())
            audio_adj = build_knn_neighbourhood(audio_adj, topk=self.audio_knn_k)
            audio_adj = compute_normalized_laplacian(audio_adj).cpu()

            self.audio_original_adj = (1- self.alpha) * self.audio_original_adj + self.alpha * audio_adj

            self.audio_original_adj = self.audio_original_adj.to_sparse_coo()
            del audio_adj

            torch.cuda.empty_cache()
            self.image_original_adj = self.image_original_adj.to(self.device)
            self.text_original_adj = self.text_original_adj.to(self.device)
            self.audio_original_adj = self.audio_original_adj.to(self.device)


    # 每个epoch会调用一次
    def pre_epoch_processing(self):
        # 训练的时候 每个epoch 重新随机采样10%的物品模拟模态缺失的状态，并且重新更新边的信息
        all_items = np.arange(self.n_items)

        train_miss_items = np.setdiff1d(all_items, self.test_miss_items)

        n_missing = int(len(train_miss_items) * 0.1)

        train_missing_items = np.random.choice(train_miss_items, n_missing, replace=False)

        missing_type = np.random.randint(0, 4, size=n_missing)

        train_visual_missing_mask = np.ones(self.n_items, dtype=np.int32)
        train_textual_missing_mask = np.ones(self.n_items, dtype=np.int32)
        train_audio_missing_mask = np.ones(self.n_items, dtype=np.int32)

        train_visual_missing_mask[train_missing_items[missing_type == 0]] = 0
        train_textual_missing_mask[train_missing_items[missing_type == 1]] = 0
        train_audio_missing_mask[train_missing_items[missing_type == 2]] = 0

        train_visual_missing_mask[train_missing_items[missing_type == 3]] = 0
        train_textual_missing_mask[train_missing_items[missing_type == 3]] = 0
        train_audio_missing_mask[train_missing_items[missing_type == 3]] = 0


        self.train_visual_missing_mask = torch.tensor(train_visual_missing_mask).to(self.device).int()
        self.train_textual_missing_mask = torch.tensor(train_textual_missing_mask).to(self.device).int()
        self.train_audio_missing_mask = torch.tensor(train_audio_missing_mask).to(self.device).int()

        print("Train - Total available items {} V missing: {}, T missing: {}, A missing: {}".format( n_missing, (self.train_visual_missing_mask == 0).sum(), (self.train_textual_missing_mask == 0).sum(), (self.train_audio_missing_mask == 0).sum()))

        if self.epoch_counter % 20 == 0 and self.epoch_counter > 0:
            self.update_adj()
        self.epoch_counter += 1
        

   
    def get_z_star(self, prior_mean, prior_logvar, means, logvars, missing_mask):
        """
        Product-of-Experts with a persistent prior expert.

        Args:
            prior_mean:   [B, D], mean of prior expert
            prior_logvar: [B, D], log variance of prior expert
            means:        list of [B, D], modality posterior means
            logvars:      list of [B, D], modality posterior log variances
            missing_mask: [B, M] or None
                        1 means observed modality expert is available
                        0 means modality expert is excluded from PoE

        Returns:
            z_mean: [B, D]
            z_var:  [B, D]
        """
        batch_size, dim = prior_mean.shape
        device = prior_mean.device
        dtype = prior_mean.dtype

        # prior expert: always active
        prior_var = torch.exp(prior_logvar).clamp(min=1e-8)
        inverse_var = 1.0 / prior_var
        inverse_var_weighted_mean = prior_mean / prior_var

        num_mods = len(means)

        for i in range(num_mods):
            mean_i = means[i].to(dtype)
            var_i = torch.exp(logvars[i]).to(dtype).clamp(min=1e-8)

            m = missing_mask[:, i:i+1].to(dtype)

            inverse_var = inverse_var + m / var_i
            inverse_var_weighted_mean = inverse_var_weighted_mean + m * mean_i / var_i

        inverse_var = inverse_var.clamp(min=1e-8)
        z_var = 1.0 / inverse_var
        z_mean = inverse_var_weighted_mean * z_var

        return z_mean, z_var

    def flow_matching_generation(self, item_id_embeds, item_history_user_embeds, image_embeds, text_embeds, audio_embedding, modality_missing_mask, train=False):
        """Flow Matching based Generative Process for Modality Feature Generation"""
        # Placeholder for future implementation
        original_v_feat, v_mean, v_logvar = self.vae_v_model(image_embeds)
        original_t_feat, t_mean, t_logvar = self.vae_t_model(text_embeds)
        original_a_feat, a_mean, a_logvar = self.vae_a_model(audio_embedding)


        prior_mean = item_id_embeds + self.prior_mu(item_id_embeds)
        vars = torch.ones_like(v_logvar).detach() * 10.0  
        prior_logvar = torch.log(vars)

        # prior_logvar = self.prior_logvar.expand_as(prior_mean).clamp(-4.0, 4.0)
        # set fixed variance
        # vars = torch.ones_like(v_logvar).detach() * 10.0  
        # logvars = torch.log(vars)

        items_mean = [v_mean, t_mean, a_mean] 
        items_logvar = [v_logvar, t_logvar, a_logvar]

        z_mean, z_var = self.get_z_star(prior_mean, prior_logvar, means=items_mean, logvars=items_logvar, missing_mask=modality_missing_mask)

        reparameterize_z_feat = z_mean + torch.randn_like(z_mean) * torch.sqrt(z_var)
       
        # 针对
        generated_v_feat, _ = self.z2v_flow_model.inference(reparameterize_z_feat, item_history_user_embeds)
        generated_t_feat, _ = self.z2t_flow_model.inference(reparameterize_z_feat, item_history_user_embeds)
        generated_a_feat, _ = self.z2a_flow_model.inference(reparameterize_z_feat, item_history_user_embeds)


        obs_v = modality_missing_mask[:, 0].unsqueeze(1).int()
        obs_t = modality_missing_mask[:, 1].unsqueeze(1).int()
        obs_a = modality_missing_mask[:, 2].unsqueeze(1).int()

        alpha_obs = 0.8
        alpha_miss = 0.2


        alpha_v = obs_v * alpha_obs + (1 - obs_v) * alpha_miss
        alpha_t = obs_t * alpha_obs + (1 - obs_t) * alpha_miss
        alpha_a = obs_a * alpha_obs + (1 - obs_a) * alpha_miss


        # image_conv = alpha_v * original_v_feat + (1 - alpha_v) * generated_v_feat
        # text_conv  = alpha_t * original_t_feat + (1 - alpha_t) * generated_t_feat


        if train:
            image_conv = alpha_v * original_v_feat + (1 - alpha_v) * generated_v_feat
            text_conv  = alpha_t * original_t_feat + (1 - alpha_t) * generated_t_feat
            audio_conv = alpha_a * original_a_feat + (1 - alpha_a) * generated_a_feat

        else:
            image_conv = obs_v * original_v_feat + (1 - obs_v) * generated_v_feat
            text_conv  = obs_t * original_t_feat + (1 - obs_t) * generated_t_feat
            audio_conv = obs_a * original_a_feat + (1 - obs_a) * generated_a_feat


        obs_v = modality_missing_mask[:, 0].bool()
        obs_t = modality_missing_mask[:, 1].bool()
        obs_a = modality_missing_mask[:, 2].bool()

        _, id2t_alignment_loss = self.z2v_flow_model.compute_loss(reparameterize_z_feat[obs_t], original_v_feat[obs_t], item_history_user_embeds[obs_t]) 
        _, id2v_alignment_loss = self.z2t_flow_model.compute_loss(reparameterize_z_feat[obs_v], original_t_feat[obs_v], item_history_user_embeds[obs_v])
        _, id2a_alignment_loss = self.z2a_flow_model.compute_loss(reparameterize_z_feat[obs_a], original_a_feat[obs_a], item_history_user_embeds[obs_a])


        item_alignment_loss = (id2t_alignment_loss + id2v_alignment_loss + id2a_alignment_loss) / 3

        self.trans_image_embedding.data = image_conv
        self.trans_text_embedding.data = text_conv        
        self.trans_audio_embedding.data = audio_conv

        return image_conv, text_conv, audio_conv, item_alignment_loss

    def forward(self, adj, train=False):
        

        #   User-Item (Behavioral) View
        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight
        #  User-Item GCN for Content Embedding
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]

        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        content_embeds = all_embeddings
    
        gcn_user_embs, gcn_item_embs = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)


        # start modality 
        image_feats = self.image_embedding.weight
        text_feats = self.text_embedding.weight
        audio_feats = self.audio_embedding.weight
        item_history_user_embeds = torch.sparse.mm(self.R.T, gcn_user_embs)

        if train:
            modality_missing_mask = torch.cat([self.train_visual_missing_mask.unsqueeze(1), self.train_textual_missing_mask.unsqueeze(1), self.train_audio_missing_mask.unsqueeze(1)], dim=1)
        else:
            modality_missing_mask = torch.cat([self.test_visual_missing_mask.unsqueeze(1), self.test_textual_missing_mask.unsqueeze(1), self.test_audio_missing_mask.unsqueeze(1)], dim=1)

        #   Spectrum Modality Fusion
        image_conv, text_conv, audio_conv, generation_loss = self.flow_matching_generation(item_embeds, item_history_user_embeds, image_feats, text_feats, audio_feats, modality_missing_mask, train)
        
        image_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_v(image_conv))
        text_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_t(text_conv))
        audio_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_a(audio_conv))
        

        #   Item-Item Modality Specific and Fusion views
        #   Image-view
        if self.sparse:
            for i in range(self.n_layers):
                image_item_embeds = torch.sparse.mm(self.image_original_adj, image_item_embeds)
        else:
            for i in range(self.n_layers):
                image_item_embeds = torch.mm(self.image_original_adj, image_item_embeds)
        image_user_embeds = torch.sparse.mm(self.R, image_item_embeds)
        image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)

        #   Text-view
        if self.sparse:
            for i in range(self.n_layers):
                text_item_embeds = torch.sparse.mm(self.text_original_adj, text_item_embeds)
        else:
            for i in range(self.n_layers):
                text_item_embeds = torch.mm(self.text_original_adj, text_item_embeds)
        text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
        text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)

        #  Audio-view
        if self.sparse:
            for i in range(self.n_layers):
                audio_item_embeds = torch.sparse.mm(self.audio_original_adj, audio_item_embeds)
        else:
            for i in range(self.n_layers):
                audio_item_embeds = torch.mm(self.audio_original_adj, audio_item_embeds)
        audio_user_embeds = torch.sparse.mm(self.R, audio_item_embeds)
        audio_embeds = torch.cat([audio_user_embeds, audio_item_embeds], dim=0)


        side_embeds = torch.mean(torch.stack([image_embeds, text_embeds, audio_embeds]), dim=0) 

        all_embeds = content_embeds + side_embeds

        all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)

        if train:
            return all_embeddings_users, all_embeddings_items, side_embeds, content_embeds, generation_loss

        return all_embeddings_users, all_embeddings_items

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size

        

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.reg_weight * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

 
        image_feats = self.image_embedding.weight
        text_feats = self.text_embedding.weight
        audio_feats = self.audio_embedding.weight

        obs_v = self.train_visual_missing_mask.bool()
        obs_t = self.train_textual_missing_mask.bool()
        obs_a = self.train_audio_missing_mask.bool()


        x1_batch_v_feat, v_mean, v_var = self.vae_v_model(image_feats[obs_v])
        x1_batch_t_feat, t_mean, t_var = self.vae_t_model(text_feats[obs_t])
        x1_batch_a_feat, a_mean, a_var = self.vae_a_model(audio_feats[obs_a])
        # x1_batch_id_feat, id_mean, id_logvar = self.item_id_vae(item_embeds)
        

        vae_v_loss, recon_v, kl_v = self.vae_v_model.compute_loss( image_feats[obs_v], x1_batch_v_feat, v_mean, v_var )
        vae_t_loss, recon_t, kl_t = self.vae_t_model.compute_loss( text_feats[obs_t], x1_batch_t_feat, t_mean, t_var )
        vae_a_loss, recon_a, kl_a = self.vae_a_model.compute_loss( audio_feats[obs_a], x1_batch_a_feat, a_mean, a_var )
        # _, recon_f, kl_f = self.item_id_vae.compute_loss( item_embeds, x1_batch_id_feat, id_mean, id_logvar )
        Vae_loss = vae_v_loss + vae_t_loss + vae_a_loss
       
        # modality_missing_mask = self.preprocess_missing_modal(missing_rate=0.1, train=True)
        # print(modality_missing_mask)

        ua_embeddings, ia_embeddings, side_embeds, content_embeds, generation_loss = self.forward(self.norm_adj, train=True)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss, batch_emb_loss, _ = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings,
                                                                      neg_i_g_embeddings)

        side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)
        cl_loss = self.InfoNCE(side_embeds_items[pos_items], content_embeds_items[pos_items], 0.2) + self.InfoNCE(
            side_embeds_users[users], content_embeds_user[users], 0.2)

        return batch_mf_loss, batch_emb_loss, self.cl_loss * cl_loss, self.mask_weight_f * Vae_loss, self.mask_weight_g * generation_loss


    def full_sort_predict(self, interaction):
        user = interaction[0]
        evaluation_items = interaction[2]
        # exit()
        # adjusted_missing_mask = self.adjust_evaluation_to_real_missing(evaluation_items)
        # self.update_adj(train=False)
        # print(self.evluation_modality_missing_mask)
        restore_user_e, restore_item_e = self.forward(self.norm_adj,train=False)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores