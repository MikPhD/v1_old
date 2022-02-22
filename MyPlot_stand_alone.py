import argparse
from MyPlot import Plot

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--n_epoch', help='epoch number', type=str, default="")
args = parser.parse_args()

n_epoch = args.n_epoch

MyPlot = Plot()
MyPlot.plot_results(n_epoch)