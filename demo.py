from main import parse, train, report
from src.DRHGCN.model import DRHGCN

if __name__=="__main__":
    args = parse(print_help=True)
    args.n_splits = 10
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    train(args, DRHGCN)
    # report("runs")
