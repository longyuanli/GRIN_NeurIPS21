import numpy as np
from torch import nn
import torch
from tensorboardX import SummaryWriter
from torch.distributions import kl_divergence, Normal
from dataloader import ChargedLoader, bballDataLoader
from modules import NodeEncoder, NodeDecoder
import dgl
from metrics import calc_metrics, calc_metrics_debug


class GRIN(nn.Module):
    def __init__(self, args):
        super(GRIN, self).__init__()
        self.num_vars = args.num_vars
        self.input_size = args.input_size
        self.prediction_steps = args.prediction_steps
        self.z_dim = args.z_dim
        self.h_dim = args.h_dim
        self.num_sample = args.best_N
        self.device = args.device
        self.learning_rate = args.lr
        self.num_epochs = args.num_epochs
        self.log_path = args.log_path
        self.attn_head = args.attn_head
        self.batch_size = args.batch_size
        self.trainable_sigma = args.ts
        self.sigma_x_init = args.sigma_x
        self.shuffle = args.shuffle
        print("shuffle: ", self.shuffle)
        self.writer = SummaryWriter(self.log_path / "tb_log")
        self.build()

    def build(self):
        self.encoder = NodeEncoder(self.h_dim,
                                   self.z_dim,
                                   self.num_vars).to(self.device)

        self.decoder = NodeDecoder(self.input_size,
                                   self.h_dim,
                                   self.z_dim,
                                   self.attn_head,
                                   self.num_sample).to(self.device)
        if self.trainable_sigma:
            self.sigma_x = nn.Parameter(torch.tensor(
                [self.sigma_x_init]).clamp(min=1e-3))
        else:
            self.sigma_x = torch.tensor([self.sigma_x_init]).to(self.device)
        self.to(self.device)

    def _train(self, train_dataloader, valid_dataloader):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        val_best = np.inf
        train_batch_id = 0
        val_batch_id = 0
        for e_idx in range(self.num_epochs):
            self.train()
            for past, future in train_dataloader(self.batch_size, shuffle=self.shuffle):
                train_batch_id += 1
                q_graph = self.build_graph(past, future).to(self.device)
                p_graph = self.build_graph(past).to(self.device)
                q_res = self.forward(q_graph)
                p_res = self.forward(p_graph)
                loss_dict = self.get_loss(q_graph, p_graph, q_res, p_res)
                loss = loss_dict["total_loss"]
                self.writer.add_scalar(
                    "train_loss_kl", loss_dict['loss_kl'], train_batch_id)
                self.writer.add_scalar(
                    "train_loss_nll", loss_dict['loss_nll'], train_batch_id)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print("Training data: ",
                      "\t".join([k + ":" + f"{v.item():.3f}" for k, v in loss_dict.items()]), f"sigma_x: {self.sigma_x.data[0]:.3f}")

            self.eval()
            val_loss = 0
            with torch.no_grad():
                for past, future in valid_dataloader(self.batch_size, shuffle=False):
                    val_batch_id += 1
                    q_graph = self.build_graph(past, future).to(self.device)
                    p_graph = self.build_graph(past).to(self.device)
                    q_res = self.forward(q_graph)
                    p_res = self.forward(p_graph)
                    loss_dict = self.get_loss(q_graph, p_graph, q_res, p_res)
                    print("Validation data: ",
                          "\t".join([k + ":" + f"{v.item():.3f}" for k, v in loss_dict.items()]))
                    val_loss += loss_dict["total_loss"]
                    self.writer.add_scalar(
                        "val_loss_kl", loss_dict['loss_kl'].detach().cpu().numpy(), val_batch_id)
                    self.writer.add_scalar(
                        "val_loss_nll", loss_dict['loss_nll'], val_batch_id)
            results = self._eval(valid_dataloader)
            eval_results = calc_metrics_debug(results)
            print("Evaluation results: ",
                  "\t".join([f"{k}: {v:.3f}" for k, v in eval_results.items()]))
            if val_loss < val_best:
                self.save()
                val_best = val_loss
                print("Imporve! Saving model. ")
            print("Best loss:", val_best)

    def _eval(self, test_dataloader, num_sample=100, rolling=False):
        self.eval()
        alpha = []
        results = {}
        p_predictions, q_predictions = [], []
        loc_pasts, loc_targets = [], []
        kl_avg = 0
        i = 0
        with torch.no_grad():
            for past, future in test_dataloader(batch_size=self.batch_size, shuffle=False):
                i += 1
                loc_past, vel_past = past
                loc_fur, vel_fur = future
                b_size = loc_past.shape[0]
                p_graph = self.build_graph(past).to(self.device)
                q_graph = self.build_graph(past, future).to(self.device)
                p_res = self.forward(p_graph)
                q_res = self.forward(q_graph)
                loss_dict = self.get_loss(q_graph, p_graph, q_res, p_res)
                if rolling:
                    p_prediction = self.rolling_predict(
                        p_graph, self.prediction_steps, num_sample=num_sample).detach().cpu().numpy()
                else:
                    p_prediction = self.predict(
                        p_graph, self.prediction_steps, num_sample=num_sample).detach().cpu().numpy()
                q_prediction = self.predict(
                    q_graph, self.prediction_steps, q_predict=True, num_sample=num_sample).detach().cpu().numpy()

                kl_avg += loss_dict['loss_kl'] * b_size / len(test_dataloader)
                p_predictions.append(p_prediction)
                q_predictions.append(q_prediction)
                # save learned attention graph
                alpha.append(
                    p_graph.ndata["alpha"].clone().detach().cpu().numpy())
                loc_pasts.append(loc_past)
                loc_targets.append(loc_fur)
        print('Packaging results')
        p_predictions = np.concatenate(p_predictions, axis=0)
        q_predictions = np.concatenate(q_predictions, axis=0)
        # for calc nll
        results['sigma_x'] = self.sigma_x.data.detach().cpu().numpy()
        results['kl_avg'] = kl_avg.detach().cpu().numpy()
        results['q_loc_pred'] = q_predictions[:, :, :, :, :2]

        results['alpha'] = np.concatenate(alpha, axis=0)  # for visualization
        # for calc prediction performance
        results['p_loc_pred'] = p_predictions[:, :, :, :, :2]
        results['loc_targets'] = np.concatenate(loc_targets, axis=0)
        results['loc_pasts'] = np.concatenate(loc_pasts, axis=0)
        return results

    def build_graph(self, past, future=None):
        # build a dgl graph for a batch sequence
        # future data is used to encode q(z|x^P, x^F)
        loc_past, vel_past = past
        if future:
            loc_future, vel_future = future
            loc = np.concatenate([loc_past, loc_future], axis=1)
            vel = np.concatenate([vel_past, vel_future], axis=1)
        else:
            loc = loc_past
            vel = vel_past
        data_feat = torch.Tensor(np.concatenate([loc, vel], axis=-1))

        num_vars = self.num_vars
        off_diag = np.ones([num_vars, num_vars]) - np.eye(num_vars)
        rel_src = np.where(off_diag)[0]
        rel_dst = np.where(off_diag)[1]

        # print(data_feat.shape) [bz, timesteps, num_var, feat]
        data_feat = data_feat.transpose(2, 1)

        N = data_feat.shape[0]
        graphs, labels = [], []

        for ii in range(N):
            label = None  # data_edges[ii]
            graph = dgl.graph((rel_src, rel_dst))
            graph.ndata["feat"] = data_feat[ii]
            graphs.append(graph)
            labels.append(label)

        graphs = dgl.batch(graphs)
        # return shape: [batch_size * num_var, timesteps, input_size]
        return graphs

    def forward(self, graph):
        feats = graph.ndata['feat']
        zG_rv, zA_rv = self.encoder(graph, feats)
        zG = zG_rv.rsample([self.num_sample]).transpose(1, 0)
        zA = zA_rv.rsample([self.num_sample]).transpose(1, 0)
        graph.ndata['zG'] = zG
        graph.ndata['zA'] = zA
        loc_pred = self.decoder.forward(
            graph, graph.ndata['feat'], self.prediction_steps)
        # loc_pred: shape [batch_size * num_vars, num_timesteps, num-sample, num_inputs]
        # e.g. for charged dataset with batch_size=128: [640, 100, 1, 4] when reconstructing from q(z|x_1:100)
        res = {"loc_pred": loc_pred,
               'zG_rv': zG_rv,
               'zA_rv': zA_rv,
               'zG': zG,
               'zA': zA}
        return res

    def get_loss(self, q_graph, p_graph, q_res, p_res):
        q_loc_pred = q_res["loc_pred"][:, :-1, :, :]  # stagger the first
        q_target = q_graph.ndata['feat'][:, 1:, :]
        q_zA_rv, q_zG_rv = q_res['zA_rv'], q_res['zG_rv']
        p_zA_rv, p_zG_rv = p_res['zA_rv'], p_res['zG_rv']
        predict_rv = Normal(q_loc_pred, self.sigma_x)
        loss_nll = - predict_rv.log_prob(torch.stack([q_target] * self.num_sample, dim=2)).reshape(
            -1, self.num_vars, q_loc_pred.shape[1], q_loc_pred.shape[2], q_loc_pred.shape[3])  # .sum([1, 2]).mean() # add dimension check
        kl_zG = kl_divergence(q_zG_rv, p_zG_rv).reshape(-1,
                                                        self.num_vars, self.z_dim)
        kl_zA = kl_divergence(q_zA_rv, p_zA_rv).reshape(-1,
                                                        self.num_vars, self.z_dim)
        # loss_nll shape (batch_size, num_vars, num_timestampes, best_N, 4)
        #                ([128, 11, 49, 4, 4])
        # kl_zG and kl_zA shape (batch_size, num_vars, z_dim)
        #                       ([128, 11, 2])
        b_s = torch.sum(loss_nll, (1, 2, 4))
        loss_nll = torch.min(b_s, dim=1)[0].mean() / self.num_vars
        loss_kl = kl_zG.sum(2).mean() + kl_zA.sum(2).mean()
        loss = loss_nll + loss_kl
        return {"total_loss": loss.mean(),
                "loss_nll": loss_nll.mean(),
                "loss_kl": loss_kl.mean(),
                'kl_zA': kl_zA.mean(),
                'kl_zG': kl_zG.mean()
                }

    def predict(self, graph, prediction_steps, q_predict=False, num_sample=None):
        # q_predict: use the z encoded by full sequence to predict the future trajs
        if num_sample is None:
            num_sample = self.num_sample
        feats = graph.ndata["feat"]
        if not q_predict:
            data_decoder = feats[:, -1:, :]
        else:
            data_decoder = feats[:, -(self.prediction_steps + 1)
                                      :-self.prediction_steps, :]
        # encode zA and zG given past sequence using prior encoder
        zG_rv, zA_rv = self.encoder(graph, feats)

        zG = zG_rv.sample([num_sample]).transpose(1, 0)
        zA = zA_rv.sample([num_sample]).transpose(1, 0)
        graph.ndata['zA'], graph.ndata['zG'] = zA, zG

        output = self.decoder.forward(
            graph, data_decoder, self.prediction_steps, forecast=True)
        output = output.reshape(-1, self.num_vars,
                                prediction_steps, num_sample, self.input_size).transpose(2, 1)
        # output shape: [batch_size, num_vars, prediction_steps, num_sample, input_size]
        # e.g. for charged balls with batch_size=128, sample 100 times: shape = [128, 5, 20, 100, 4]
        return output

    def save(self):
        path = self.log_path / "best_model.pt"
        torch.save(self.state_dict(), path)

    def load(self):
        path = self.log_path / "best_model.pt"
        self.load_state_dict(torch.load(path, map_location=self.device))


if __name__ == '__main__':
    from multiprocessing import Pool
    import numpy as np
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode", choices=['train', 'eval'], help='runing mode : (train, eval)')
    parser.add_argument(
        "--dataset", choices=['charged', 'bball'], help='runing mode : (charged, bball)', default="charged")
    parser.add_argument("--seed", default=0,
                        help="the seed for reproducing experiment results")
    parser.add_argument("--num_sample", default=100, type=int,
                        help="num_samples for prediction")
    parser.add_argument("--data_idx", default=1, type=int,
                        help="num_samples for prediction")

    # Network-related parameters
    parser.add_argument("--task_name", default="task_0",
                        help="a task name to indicate the log path and model save path")
    parser.add_argument("--z_dim", default=2, type=int,
                        help='dimension for the dimension of stochastic latent variable z_A and z_G (default: 2).')
    parser.add_argument("--h_dim", default=64, type=int,
                        help='dimension for the hidden layers of MLPs (default: 128).')
    parser.add_argument("--best_N", default=1, type=int,
                        help='the used N for best-of-N losses used in training (default: 1).')
    parser.add_argument("--attn_head", default=2, type=int,
                        help='number of attention heads (default: 1).')

    # Training-related paramters
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--log_path", default=None,
                        help="Path to save the tensorboard log and model")
    parser.add_argument("--num_epochs", default=400,
                        type=int, help='maximum number of epochs')
    parser.add_argument("--lr", default=1e-3, type=float,
                        help="learning rate in optimizer")
    parser.add_argument("--ts", action="store_true",
                        help="is trainable sigma_x?")
    parser.add_argument("--shuffle", action="store_true",
                        help="shuffle the train data?")
    parser.add_argument("--sigma_x", default=1e-1, type=float,
                        help="sigma_x init value")

    args = parser.parse_args()

    # Fixed parameters for charged particles dataset
    if args.dataset == 'charged':
        args.num_vars = 5
        args.input_size = 4
        args.prediction_steps = 20
        args.train_len = 80
        train_dataloader = ChargedLoader(
            "dataset/charged/train_big.npz", args.train_len)
        valid_dataloader = ChargedLoader(
            "dataset/charged/valid.npz", args.train_len)
        test_dataloader = ChargedLoader(
            "dataset/charged/test.npz", args.train_len)
    if args.dataset == 'bball':
        args.num_vars = 11
        args.input_size = 4
        args.prediction_steps = 10
        args.train_len = 40
        train_dataloader = bballDataLoader(
            "dataset/bball/train.npy", args.train_len)
        valid_dataloader = bballDataLoader(
            "dataset/bball/val.npy", args.train_len)
        test_dataloader = bballDataLoader(
            "dataset/bball/test.npy", args.train_len)
    args.log_path = Path('./logs') / args.task_name
    args.log_path.mkdir(exist_ok=True, parents=True)
    print(args)
    model = GRIN(args)

    if args.mode == 'train':
        model._train(train_dataloader, valid_dataloader)

    if args.mode == 'eval':
        model.load()
        train_dataloader = bballDataLoader(
            "dataset/bball/train.npy", args.train_len, num_sample=10000)
        results = model._eval(test_dataloader, num_sample=args.num_sample)
        eval_results = calc_metrics(results)
        import pickle
        pickle.dump(results, open(
            f"./{args.task_name}.pkl", 'wb'))
        print("Evaluation: ",
              "\t".join([f"{k}: {v:.3f}" for k, v in eval_results.items()]))
