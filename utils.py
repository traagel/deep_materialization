import random
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import torch as th
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ogb.linkproppred import Evaluator


def norm_plot(curves, title):
    """
    Plots normal distribution curves
    curves: list of tuples like: (mu, sigma, label)
    """
    fig, ax = plt.subplots()
    for mu, sigma, label in curves:
        x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), label=label)

    ax.set_title(title)
    ax.legend()


class Logger(object):
    """
    Note that this code is copy/pasted from the OGB PyG implementation
    for linkproppred in DDI, so as to ensure we measure/report performance
    in the same way.

    Link:
    https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/ddi/logger.py
    """

    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * th.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f"Run {run + 1:02d}:")
            print(f"Highest Train: {result[:, 0].max():.2f}")
            print(f"Highest Valid: {result[:, 1].max():.2f}")
            print(f"  Final Train: {result[argmax, 0]:.2f}")
            print(f"   Final Test: {result[argmax, 2]:.2f}")
        else:
            result = 100 * th.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = th.tensor(best_results)

            print(f"All runs:")
            r = best_result[:, 0]
            print(f"Highest Train: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 1]
            print(f"Highest Valid: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 2]
            print(f"  Final Train: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 3]
            print(f"   Final Test: {r.mean():.2f} ± {r.std():.2f}")


class NegativeSamplerFast(object):
    def __init__(self, g):
        """
        Ultimately want to get the row/column pairs of the adjacency matrix
        that are equal to 0, since these represent the edges that do not exist.
        We will then randomly sample from these edges.
        """
        # Get 1's where there's zeros (except the diagonal) in the
        # adjacency matrix
        # i.e., 1 - (A + I)
        invA = 1 - g.add_self_loop().adj().to_dense()
        # Extract the rows/columns of the entries that are non-zero
        # i.e., the source/destination nodes of all the negative edges
        u, v = th.where(invA)
        # Convert each to list for faster sampling
        u = u.tolist()
        v = v.tolist()
        self.neg_edges = list(zip(u, v))

    def __call__(self, N):
        return th.tensor(random.choices(self.neg_edges, k=N)).T


class SAGE(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dropout,
        sage_cls,
    ):
        super(SAGE, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(sage_cls(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(sage_cls(hidden_channels, hidden_channels))
        self.convs.append(sage_cls(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    @property
    def num_layers(self):
        return len(self.convs)

    def forward(self, gs, x):
        h = x

        for i, conv in enumerate(self.convs):
            if gs.is_block:
                g = gs[i]
            else:
                g = gs  # full graph
            h = conv(g, h)
            if i != len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        return h


def train_epoch(
    g, emb, model, predictor, split_edge, optimizer, neg_sampler, device
):

    emb = emb.train()
    model = model.train()
    predictor = predictor.train()

    emb = emb.to(device)
    model = model.to(device)
    predictor = predictor.to(device)

    pos_train_edge = split_edge["train"]["edge"].to(emb.weight.device)

    logging = dict()

    total_loss = total_examples = 0
    for perm in DataLoader(
        range(pos_train_edge.size(0)), 64 * 1024, shuffle=True
    ):

        optimizer.zero_grad()

        h = model(g.to(device), emb.weight)

        edge = pos_train_edge[perm].t()
        u, v = edge

        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -th.log(pos_out + 1e-15).mean()

        edge_neg = neg_sampler(u.shape[0])

        neg_out = predictor(h[edge_neg[0]], h[edge_neg[1]])
        neg_loss = -th.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        th.nn.utils.clip_grad_norm_(emb.weight, 1.0)
        th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        th.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        logging.update(
            dict(
                bs=perm.shape[0],
                pos_loss=pos_loss.item(),
                neg_loss=neg_loss.item(),
                loss=loss.item(),
            )
        )

        # print(logging)
        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@th.no_grad()
def test(emb, model, predictor, g, split_idx, device):
    emb = emb.eval()
    model = model.eval()
    predictor = predictor.eval()

    evaluator = Evaluator(name="ogbl-ddi")

    # Extract node embeddings
    h = model(g.to(device), emb.weight)
    h = h.to("cpu")
    predictor = predictor.to("cpu")

    pos_train_pred = predictor(
        h[split_idx["eval_train"]["edge"][:, 0], :],
        h[split_idx["eval_train"]["edge"][:, 1], :],
    ).flatten()
    pos_valid_pred = predictor(
        h[split_idx["valid"]["edge"][:, 0], :],
        h[split_idx["valid"]["edge"][:, 1], :],
    ).flatten()
    neg_valid_pred = predictor(
        h[split_idx["valid"]["edge_neg"][:, 0], :],
        h[split_idx["valid"]["edge_neg"][:, 1], :],
    ).flatten()
    pos_test_pred = predictor(
        h[split_idx["test"]["edge"][:, 0], :],
        h[split_idx["test"]["edge"][:, 1], :],
    ).flatten()
    neg_test_pred = predictor(
        h[split_idx["test"]["edge_neg"][:, 0], :],
        h[split_idx["test"]["edge_neg"][:, 1], :],
    ).flatten()

    results = {}
    for K in [10, 20, 30]:
        evaluator.K = K
        train_hits = evaluator.eval(
            {
                "y_pred_pos": pos_train_pred,
                "y_pred_neg": neg_valid_pred,
            }
        )[f"hits@{K}"]
        valid_hits = evaluator.eval(
            {
                "y_pred_pos": pos_valid_pred,
                "y_pred_neg": neg_valid_pred,
            }
        )[f"hits@{K}"]
        test_hits = evaluator.eval(
            {
                "y_pred_pos": pos_test_pred,
                "y_pred_neg": neg_test_pred,
            }
        )[f"hits@{K}"]

        results[f"Hits@{K}"] = (train_hits, valid_hits, test_hits)

    return results


def train(g, emb, model, predictor, split_edge, device, loggers, args, run):
    """
    A complete model training loop
    """
    optimizer = th.optim.Adam(
        list(model.parameters())
        + list(emb.parameters())
        + list(predictor.parameters()),
        lr=args["lr"],
    )
    neg_sampler = NegativeSamplerFast(g)

    for epoch in range(1, 1 + args["epochs"]):
        loss = train_epoch(
            g,
            emb,
            model,
            predictor,
            split_edge,
            optimizer,
            neg_sampler,
            device,
        )

        if epoch % args["eval_steps"] == 0:
            results = test(emb, model, predictor, g, split_edge, device)

            for key, result in results.items():
                loggers[key].add_result(run, result)

            if epoch % args["log_steps"] == 0:
                for key, result in results.items():
                    train_hits, valid_hits, test_hits = result
                    print(key)
                    print(
                        f"Run: {run+1:02d}, "
                        f"Epoch: {epoch:02d}, "
                        f"Loss: {loss:.4f}, "
                        f"Train: {100 * train_hits:.2f}%, "
                        f"Valid: {100 * valid_hits:.2f}%, "
                        f"Test: {100 * test_hits:.2f}%"
                    )
                print("---")

    return loggers


def repeat_experiments(
    g, emb, model, predictor, split_idx, device, train_args, n_runs
):
    loggers = {
        "Hits@10": Logger(n_runs, train_args),
        "Hits@20": Logger(n_runs, train_args),
        "Hits@30": Logger(n_runs, train_args),
    }
    for run in range(n_runs):
        th.nn.init.xavier_uniform_(emb.weight)
        model.reset_parameters()
        predictor.reset_parameters()

        loggers = train(
            g,
            emb,
            model,
            predictor,
            split_idx,
            device,
            loggers,
            train_args,
            run,
        )

        # Print run stats
        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

    # Print experiment stats
    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()

    return loggers
