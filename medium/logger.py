import torch


class Logger(object):
    """ Adapted from https://github.com/snap-stanford/ogb/ """

    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 4
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None, mode='max_acc', wandb=None,run_result=None):
        if run is not None:
            result = torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item() # 选择valid指标最高的，对于有valid集的，应该选这个，默认也是这个
            argmin = result[:, 3].argmin().item() #选择loss最小的，evaluate_batch全是0,所以不能选这个策略
            if mode == 'max_acc':
                ind = argmax
            else:
                ind = argmin
            print_str = f'Run {run + 1:02d}:' + \
                        f'Highest Train: {result[:, 0].max():.2f} ' + \
                        f'Highest Valid: {result[:, 1].max():.2f} ' + \
                        f'Highest Test: {result[:, 2].max():.2f}\n' + \
                        f'Chosen epoch: {ind} ' + \
                        f'Final Train: {result[ind, 0]:.2f} ' + \
                        f'Final Test: {result[ind, 2]:.2f}'
            print(print_str)
            self.test = result[ind, 2]
            if wandb is not None:
                wandb.log({"Run_{}_Train".format(run): result[:, 0].max()})
                wandb.log({"Run_{}_Valid".format(run): result[:, 1].max()})
                wandb.log({"Run_{}_Test".format(run): result[:, 2].max()})
                wandb.log({"Run_{}_Chosen_epoch".format(run): ind + 1})
                wandb.log({"Run_{}_Final_Train".format(run): result[ind, 0]*100})
                wandb.log({"Run_{}_Final_Test".format(run): result[ind, 2]*100})
                if run_result is not None:
                    run_result.append(result[ind, 2].item()*100)
        else:
            best_results = []
            max_val_epoch = 0
            for r in self.results:
                r = 100 * torch.tensor(r)
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                test1 = r[:, 2].max().item()
                if mode == 'max_acc':
                    train2 = r[r[:, 1].argmax(), 0].item()
                    test2 = r[r[:, 1].argmax(), 2].item()
                    max_val_epoch = r[:, 1].argmax()
                else:
                    train2 = r[r[:, 3].argmin(), 0].item()
                    test2 = r[r[:, 3].argmin(), 2].item()
                best_results.append((train1, test1, valid, train2, test2))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Test: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 4]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            self.test = r.mean()
            if wandb is not None:
                wandb.log({"Highest_Train": best_result[:, 0].mean()})
                wandb.log({"Highest_Train_std": best_result[:, 0].std()})
                wandb.log({"Highest_Test": best_result[:, 1].mean()})
                wandb.log({"Highest_Test_std": best_result[:, 1].std()})
                wandb.log({"Highest_Valid": best_result[:, 2].mean()})
                wandb.log({"Highest_Valid_std": best_result[:, 2].std()})
                wandb.log({"Final_Train": best_result[:, 3].mean()})
                wandb.log({"Final_Train_std": best_result[:, 3].std()})
                wandb.log({"Final_Test": best_result[:, 4].mean()})
                wandb.log({"Final_Test_std": best_result[:, 4].std()})



import os


def save_result(args, results):
    if not os.path.exists(f'results/{args.dataset}'):
        os.makedirs(f'results/{args.dataset}')
    filename = f'results/{args.dataset}/{args.method}.csv'
    print(f"Saving results to {filename}")
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(
            f"{args.method} " + f"{args.kernel}: " + f"{args.weight_decay} " + f"{args.dropout} " + \
            f"{args.num_layers} " + f"{args.alpha}: " + f"{args.hidden_channels}: " + \
            f"{results.mean():.2f} $\pm$ {results.std():.2f} \n")
