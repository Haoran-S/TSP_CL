from generate_data import load_datasets
from model.common import SumRateLoss
import seaborn as sns
import importlib
import matplotlib.pyplot as plt
import torch
import argparse

parser = argparse.ArgumentParser(description='generate figure')
parser.add_argument('--ext', type=str, default='')
args0 = parser.parse_args()
markerseq = ['x', 'None', 's', 'o', '*']
seq0 = ['results/single_online', 'results/single_joint',
        'results/reservoir_sampling_online', 'results/minmax_online']
leg = ['TL', 'Joint', 'Reservoir', 'MinMax (Proposed)']
seq = [i+args0.ext+'.pt' for i in seq0]

data = [torch.load(seq[i]) for i in range(len(seq))]
num_methods = len(data)
num_tasks = len(data[0][0][0])
arg = data[0][6]
episode_lens = torch.load(arg.data_file)[2].num_train
episode_lens = [int(k) for k in episode_lens.split('-')]
print('episode size:', episode_lens)
noise_var = torch.load(arg.data_file)[2].noise
interval = int(episode_lens[0] * arg.n_epochs / arg.batch_size / arg.log_every)
bs = arg.batch_size * arg.log_every / 1000


# rate per task
fig, axs = plt.subplots(nrows=num_tasks, ncols=1,
                        sharex=True, sharey=False,  figsize=(6, 6))
for t in range(num_tasks):
    plt_max = 0
    plt_min = 100
    for i in range(num_methods):
        mse = [d[t] for d in data[i][1]]
        axs[t].plot([(x+1)*bs for x in range(len(mse))],
                    mse, marker=markerseq[i])
        plt_max = max(max(mse).item(), plt_max)
        plt_min = min(min(mse).item(), plt_min)

    if args0.ext != '_unbalance':
        for k in range(num_tasks):
            axs[t].plot([interval*(k+1)*bs, interval*(k+1)*bs],
                        [plt_min, plt_max], color='lightgray', linewidth=1)
    else:
        axs[t].plot([interval*bs, interval*bs],
                    [plt_min, plt_max], color='lightgray', linewidth=1)

    axs[t].set_ylabel('Episode %d' % (t+1))
plt.legend(leg, loc='lower left', prop={'size': 10})
axs[-1].set_xlabel('number of samples seen in data stream (k)')
# plt.tight_layout()
plt.savefig('results/rate_per_task' + args0.ext + '.pdf')

# ratio per task
fig, axs = plt.subplots(nrows=num_tasks, ncols=1,
                        sharex=True, sharey=False,  figsize=(6, 6))
for t in range(num_tasks):
    plt_max = 0
    plt_min = 100
    for i in range(num_methods):
        mse = [d[t] for d in data[i][2]]
        axs[t].plot([(x+1)*bs for x in range(len(mse))],
                    mse, marker=markerseq[i])
        plt_max = max(max(mse).item(), plt_max)
        plt_min = min(min(mse).item(), plt_min)
    if args0.ext != '_unbalance':
        for k in range(num_tasks):
            axs[t].plot([interval*(k+1)*bs, interval*(k+1)*bs],
                        [plt_min, plt_max], color='lightgray', linewidth=1)
    else:
        axs[t].plot([interval*bs, interval*bs],
                    [plt_min, plt_max], color='lightgray', linewidth=1)

    axs[t].set_ylabel('Episode %d' % (t+1))
plt.legend(leg, loc='lower left', prop={'size': 10})
axs[-1].set_xlabel('number of samples seen in data stream (k)')
# plt.tight_layout()
plt.savefig('results/ratio_per_task' + args0.ext + '.pdf')


# rate mean
fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True,
                        sharey=True,  figsize=(6, 6))
plt_max = 0
plt_min = 100
for i in range(num_methods):
    mse = [torch.mean(d) for d in data[i][1]]
    axs.plot([(x+1)*bs for x in range(len(mse))], mse, marker=markerseq[i])
    plt_max = max(max(mse).item(), plt_max)
    plt_min = min(min(mse).item(), plt_min)
if args0.ext != '_unbalance':
    for i in range(num_tasks):
        axs.plot([interval*(i+1)*bs, interval*(i+1)*bs],
                 [plt_min, plt_max], color='lightgray', linewidth=1)
else:
    axs.plot([interval*bs, interval*bs],
             [plt_min, plt_max], color='lightgray', linewidth=1)


plt.legend(leg)
axs.set_ylabel('average sum-rate (bit/sec.)')
axs.set_xlabel('number of samples seen in data stream (k)')
# plt.tight_layout()
plt.savefig('results/rate_mean' + args0.ext + '.pdf', facecolor='w',
            edgecolor='w', transparent=True)

# ratio mean
fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True,
                        sharey=True,  figsize=(6, 6))
plt_max = 0
plt_min = 100
for i in range(num_methods):
    mse = [torch.mean(d) for d in data[i][2]]
    axs.plot([(x+1)*bs for x in range(len(mse))], mse, marker=markerseq[i])
    plt_max = max(max(mse).item(), plt_max)
    plt_min = min(min(mse).item(), plt_min)

if args0.ext != '_unbalance':
    for i in range(num_tasks):
        axs.plot([interval*(i+1)*bs, interval*(i+1)*bs],
                 [plt_min, plt_max], color='lightgray', linewidth=1)
else:
    axs.plot([interval*bs, interval*bs],
             [plt_min, plt_max], color='lightgray', linewidth=1)

plt.legend(leg)
axs.set_ylabel('sum-rate approximation ratio')
axs.set_xlabel('number of samples seen in data stream (k)')
# plt.tight_layout()
plt.savefig('results/ratio_mean' + args0.ext + '.pdf', facecolor='w',
            edgecolor='w', transparent=True)


def test(model, tasks, args):
    model.eval()
    MSE_per_sample = []
    SUM_per_sample = []
    ratio_per_sample = []
    for i, task in enumerate(tasks):
        t = i
        xb = task[1]
        yb = task[2]
        if args.cuda:
            xb = xb.cuda()
        output = model(xb, t).data.cpu()

        predict_sumrate = SumRateLoss(
            xb.cpu(), output, args.noise, persample=True)
        predict_mse = torch.sum((yb.cpu() - output) ** 2, 1)
        label_sumrate = SumRateLoss(
            xb.cpu(), yb.cpu(), args.noise, persample=True)
        ratio = torch.div(predict_sumrate, label_sumrate)
        if i == 0:
            ratio_per_sample = ratio
            MSE_per_sample = predict_mse
            SUM_per_sample = predict_sumrate
        else:
            ratio_per_sample = torch.cat((ratio_per_sample, ratio), 0)
            MSE_per_sample = torch.cat((MSE_per_sample, predict_mse), 0)
            SUM_per_sample = torch.cat((SUM_per_sample, predict_sumrate), 0)
    return ratio_per_sample, MSE_per_sample, -SUM_per_sample


if __name__ == "__main__":
    result_ratio = []
    result_mse = []
    result_rate = []

    for item in seq:
        out = torch.load(item)
        args = out[6]
        args.noise = noise_var
        state_dict = out[5]
        x_tr, x_te, n_inputs, n_outputs, n_tasks = load_datasets(args)
        Model = importlib.import_module('model.' + args.model)
        model = Model.Net(n_inputs, n_outputs, n_tasks, args)
        model.load_state_dict(state_dict)
        ratio_per_sample, MSE_per_sample, SUM_per_sample = test(
            model, x_te, args)
        result_ratio.append(ratio_per_sample)
        result_mse.append(MSE_per_sample)
        result_rate.append(SUM_per_sample)

    index = [(0, 0), (0, 1), (1, 0), (1, 1)]
    ls = ['-.', ':', '--', '-', '--']

    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False,
                            sharey=True,  figsize=(6, 6))
    for i in range(4):
        sns.distplot(result_ratio[i], bins=100,
                     ax=axs[index[i][0], index[i][1]], kde=False)
        axs[index[i][0], index[i][1]].title.set_text(leg[i])
        axs[index[i][0], index[i][1]].set_xlim(0, 2)
        axs[index[i][0], index[i][1]].set_yscale('log')
        axs[index[i][0], index[i][1]].set_xlabel(
            'sum-rate approximation ratios')
        axs[index[i][0], index[i][1]].set_ylabel('Density')
    plt.tight_layout()
    plt.savefig('results/ratio_pdf' + args0.ext + '.pdf',
                facecolor='w', edgecolor='w', transparent=True)

    fig, axs = plt.subplots(nrows=1, ncols=1, sharex=False,
                            sharey=True,  figsize=(6, 6))
    for i in range(4):
        kwargs = {'cumulative': True, 'linestyle': ls[i]}
        sns.distplot(result_ratio[i], bins=200,
                     hist_kws=kwargs, kde_kws=kwargs, hist=False)
    plt.legend(leg, loc='upper left')
    plt.xlim(0, 1.25)
    axs.set_xlabel('sum-rate approximation ratios')
    axs.set_ylabel('Probability')
    plt.tight_layout()
    plt.savefig('results/ratio_cdf' + args0.ext + '.pdf',
                facecolor='w', edgecolor='w', transparent=True)
