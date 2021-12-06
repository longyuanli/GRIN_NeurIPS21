from dgl.convert import graph
import torch.nn.functional as F
from torch import nn
from torch.distributions import kl_divergence, Normal
from dgl.nn.pytorch.conv import GraphConv
import torch


class GraphAttentionConv(nn.Module):
    def __init__(self, h_dim, z_dim, attn_head, K=1):
        super(GraphAttentionConv, self).__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.K = K
        self.attn_head = attn_head
        self.attn_s = nn.ModuleList()
        self.attn_d = nn.ModuleList()
        for k in range(K):
            self.attn_s.append(nn.Linear(self.z_dim, h_dim, bias=False))
            self.attn_d.append(nn.Linear(self.z_dim, h_dim, bias=False))

        self.F1 = nn.Sequential(nn.Linear(self.h_dim * self.K, self.h_dim, bias=False),
                                nn.ReLU(),
                                nn.Linear(self.h_dim, self.h_dim, bias=False))
        self.F2 = nn.Sequential(nn.Linear(2 * h_dim, self.h_dim, bias=False),
                                nn.ReLU(),
                                nn.Linear(self.h_dim, self.h_dim, bias=False))

    def edge_attention(self, edges):
        a = []
        for k in range(self.K):
            a_s = self.attn_s[k](edges.src['zG'])
            a_d = self.attn_d[k](edges.dst['zG'])
            a.append(F.leaky_relu((a_s * a_d).sum(-1).unsqueeze(-1)))

        return {'eG': torch.cat(a, dim=-1)}

    def message_func(self, edges):
        dA = self.F2(
            torch.cat([edges.src['xt_enc'], edges.dst['xt_enc']], dim=-1))

        return {'dA': dA, 'eG': edges.data['eG']}

    def reduce_func(self, nodes):

        # calculate attention weight
        alpha = F.softmax(nodes.mailbox['eG'], dim=1)
        res = []
        for k in range(self.K):
            res.append(torch.mean(alpha[:, :, :, k].unsqueeze(
                -1) * nodes.mailbox['dA'], dim=1))
        deltax = self.F1(torch.cat(res, dim=-1))
        return {'deltax': deltax, "alpha": alpha}

    def attn(self, graph):
        graph.apply_edges(self.edge_attention)
        return graph

    def conv(self, graph):
        graph.update_all(self.message_func, self.reduce_func)
        return graph


class NodeEncoder(nn.Module):
    def __init__(self, h_dim, z_dim, num_vars):
        super(NodeEncoder, self).__init__()
        self.rnn = nn.RNN(4, h_dim, 1)
        self.gcn = GraphConv(h_dim, h_dim)
        self.num_vars = num_vars
        self.zG_mu_enc = nn.Linear(h_dim, z_dim)
        self.zG_std_enc = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())
        self.zA_mu_enc = nn.Linear(h_dim, z_dim)
        self.zA_std_enc = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, graph, inputs):
        # Input shape: [batchsize * num_atoms, num_timesteps, num_dims]
        x = inputs.transpose(1, 0)
        _, x = self.rnn(x)  # Encoder Eq. 1
        x = x.squeeze(0)
        x = self.gcn(graph, x)

        zG_mu = self.zG_mu_enc(x)  # Encoder Eq. 5
        zG_std = self.zG_std_enc(x)
        zA_mu = self.zA_mu_enc(x)  # Encoder Eq. 6
        zA_std = self.zA_std_enc(x)
        return Normal(zG_mu, zG_std), Normal(zA_mu, zA_std)

    # todo use only 1 function
    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = torch.cat([h_u, h_v], 1)
        if self.mlp_in_apply_edges:
            score = self.mlp2(score)
        return {'score': score}


class NodeDecoder(nn.Module):
    def __init__(self, n_in_node, h_dim, z_dim, attn_head, num_sample):
        super(NodeDecoder, self).__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.x_enc = nn.Linear(n_in_node, h_dim)
        self.out_fc = nn.Sequential(nn.Linear(h_dim + self.z_dim, h_dim),
                                    nn.ReLU(),
                                    nn.Linear(h_dim, 4))
        self.attn_head = attn_head
        self.num_sample = num_sample

        self.GAT = GraphAttentionConv(
            self.h_dim, self.z_dim, self.attn_head, K=self.attn_head)
        print('Using learned interaction net decoder.')

    def apply_edges1(self, edges):
        h_u = edges.src['hA']
        h_v = edges.dst['hA']
        score = torch.cat([h_u, h_v], -1)
        return {'score': score}

    def reduce_func1(self, nodes):
        # the last dim always has size 1
        return {'ft': torch.mean(nodes.mailbox['m'], dim=1)}

    def single_step_forward(self, graph):
        graph.ndata['xt_enc'] = self.x_enc(graph.ndata['xt'])
        graph = self.GAT.conv(graph)
        h = torch.cat([graph.ndata['deltax'], graph.ndata['zA']], dim=-1)
        deltax = self.out_fc(h)
        graph.ndata['xt'] = graph.ndata['xt'] + deltax
        return graph

    def forward(self, graph, inputs, pred_steps, forecast=False):
        # forecast: predict the future
        # not forecast: reconstruct the input sequence
        # NOTE: Assumes that we have the same graph across all samples.
        graph = self.GAT.attn(graph)
        inputs = inputs.transpose(0, 1).contiguous()
        num_sample = graph.ndata['zA'].shape[1]
        inputs = torch.stack([inputs] * num_sample, dim=2)
        if forecast:
            # TODO check case
            sizes = [pred_steps, inputs.shape[1],
                     inputs.shape[2], inputs.shape[3]]
        else:
            sizes = inputs.shape

        preds_out = torch.zeros(sizes).to(inputs.device)
        # Only take n-th timesteps as starting points (n: pred_steps)
        last_pred = inputs[0::pred_steps, :, :]
        for b_idx in range(0, last_pred.shape[0]):
            graph.ndata['xt'] = last_pred[b_idx]
            for step in range(0, pred_steps):
                graph = self.single_step_forward(graph)
                preds_out[step + b_idx * pred_steps, :, :] = graph.ndata['xt']

        return preds_out.transpose(0, 1).contiguous()
