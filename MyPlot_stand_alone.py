import argparse
from MyPlot import Plot

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--n_epoch', help='epoch number', type=str, default="")
parser.add_argument('-sn', '--set_name', help='parameter set name', type=str, default="70-20-001-001")
args = parser.parse_args()

n_epoch = args.n_epoch
set_name = args.set_name

MyPlot = Plot(set_name)
MyPlot.plot_results(n_epoch)