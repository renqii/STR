import argparse
import logging
import sys
import torch.distributed as dist
from collections.abc import MutableMapping
from logging import getLogger
import torch

from recbole.config import Config
from recbole.data import (
    create_dataset,
    data_preparation,
)
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)


def run(
    model,
    dataset,
    config_file_list=None,
    config_dict=None,
    saved=True,
    nproc=1,
    world_size=-1,
    ip="localhost",
    port="5678",
    group_offset=0,
):
    if nproc == 1 and world_size <= 0:
        res = run_recbole(
            model=model,
            dataset=dataset,
            config_file_list=config_file_list,
            config_dict=config_dict,
            saved=saved,
        )
    else:
        if world_size == -1:
            world_size = nproc
        import torch.multiprocessing as mp

        # Refer to https://discuss.pytorch.org/t/problems-with-torch-multiprocess-spawn-and-simplequeue/69674/2
        # https://discuss.pytorch.org/t/return-from-mp-spawn/94302/2
        queue = mp.get_context("spawn").SimpleQueue()

        config_dict = config_dict or {}
        config_dict.update(
            {
                "world_size": world_size,
                "ip": ip,
                "port": port,
                "nproc": nproc,
                "offset": group_offset,
            }
        )
        kwargs = {
            "config_dict": config_dict,
            "queue": queue,
        }

        mp.spawn(
            run_recboles,
            args=(model, dataset, config_file_list, kwargs),
            nprocs=nproc,
            join=True,
        )

        # Normally, there should be only one item in the queue
        res = None if queue.empty() else queue.get()
    return res


def run_recbole(
    model=None,
    dataset=None,
    config_file_list=None,
    config_dict=None,
    saved=True,
    queue=None,
):
    r"""A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
        queue (torch.multiprocessing.Queue, optional): The queue used to pass the result to the main process. Defaults to ``None``.
    """
    # configurations initialization
    config = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    init_seed(config["seed"], config["reproducibility"])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    logger.info(model)

    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    # load RM 模型
    # # SASRec beauty
    model_pth = '/data/jiarenqi/DUR/RecBole/saved/SASRec-Dec-20-2024_14-15-46.pth'
    # SASRec beauty split-by-ratio
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/SASRec-Jan-10-2025_12-15-27.pth'
    # SASRec beauty one-seq
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/SASRec-Jan-12-2025_19-40-12.pth'
    # # GCSAN
    # model_pth = '/data1/jiarenqi/SeqRec/Baseline/RecBole/saved/GCSAN-Dec-20-2024_14-15-52.pth'
    # # NARM
    # model_pth = '/data/jiarenqi/DUR/RecBole/saved/NARM-Dec-20-2024_14-16-03.pth'
    # BERT4Rec
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/BERT4Rec-Dec-20-2024_14-16-45.pth'
    # GRU4Rec
    # model_pth = '/data/jiarenqi/DUR/RecBole/saved/GRU4Rec-Dec-20-2024_12-02-52.pth'
    # GRU4Rec RM-post model
    # model_pth = '/data/jiarenqi/DUR/RecBole/saved/GRU4Rec-Dec-20-2024_17-32-59.pth'
    # SRGNN
    # model_pth = '/data/jiarenqi/DUR/RecBole/saved/SRGNN-Dec-21-2024_11-47-20.pth'
    # FEARec beauty
    # model_pth = '/data/jiarenqi/DUR/RecBole/saved/FEARec-Dec-24-2024_11-10-23.pth'
    # LightSANs
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/LightSANs-Dec-25-2024_23-32-09.pth'
    # STAMP
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/STAMP-Dec-25-2024_23-36-55.pth'
    # RepeatNet
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/RepeatNet-Dec-25-2024_23-32-57.pth'
    # SINE
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/SINE-Dec-25-2024_21-00-52.pth'
    # HGN
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/HGN-Dec-26-2024_10-39-42.pth'
    # FOSSIL
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/FOSSIL-Dec-25-2024_23-35-06.pth'
    # CORE
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/CORE-Dec-24-2024_21-36-36.pth'
    # HRM
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/HRM-Dec-25-2024_23-32-40.pth'
    # NextItNet
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/NextItNet-Dec-25-2024_23-36-04.pth'
    # SHAN
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/SHAN-Dec-24-2024_20-46-59.pth'
    # NPE
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/NPE-Dec-25-2024_23-32-26.pth'
    # Caser
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/Caser-Dec-25-2024_23-36-37.pth'
    # SASRec yelp
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/SASRec-Dec-30-2024_15-41-34.pth'
    # GRU4Rec yelp
    # model_pth = '/data/jiarenqi/DUR/RecBole/saved/GRU4Rec-Dec-31-2024_12-33-28.pth'
    # NextItNet yelp
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/NextItNet-Dec-31-2024_17-10-38.pth'
    # SRGNN yelp
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/SRGNN-Dec-31-2024_17-11-08.pth'
    # NARM yelp
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/NARM-Dec-31-2024_17-16-04.pth'
    # STAMP yelp
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/STAMP-Dec-31-2024_17-12-10.pth'
    # Caser yelp
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/Caser-Dec-31-2024_17-10-01.pth'
    # SASRec cellphone
    # model_pth ='/data/jiarenqi/Baseline/RecBole/saved/SASRec-Jan-01-2025_21-11-43.pth'
    # SASRec sports
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/SASRec-Jan-01-2025_21-44-09.pth'
    # Caser sports
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/Caser-Jan-01-2025_22-54-53.pth'
    # NextItNet sports
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/NextItNet-Jan-01-2025_22-55-06.pth'
    # SRGNNN sports
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/SRGNN-Jan-01-2025_22-55-13.pth'
    # GRU4Rec sports
    # model_pth = '/data/jiarenqi/DUR/RecBole/saved/GRU4Rec-Jan-01-2025_22-55-20.pth'
    # NARM sports
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/NARM-Jan-01-2025_22-55-27.pth'
    # STAMP sports
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/STAMP-Jan-01-2025_22-55-45.pth'
    # GRU4Rec cellphone
    # model_pth = '/data/jiarenqi/DUR/RecBole/saved/GRU4Rec-Jan-02-2025_10-56-00.pth'
    # GRU4Rec cellphone split-by-ratio
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/GRU4Rec-Jan-10-2025_17-07-27.pth'
    # GRU4Rec cellphone one-seq
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/GRU4Rec-Jan-12-2025_19-54-52.pth'
    # NARM cellphone
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/NARM-Jan-02-2025_10-56-11.pth'
    # STAMP cellphone
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/STAMP-Jan-02-2025_10-56-27.pth'
    # SRGNN cellphone
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/SRGNN-Jan-02-2025_10-55-38.pth'
    # SRGNN cellphone split-by-ratio
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/SRGNN-Jan-10-2025_16-23-57.pth'
    # NextItNet cellphone
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/NextItNet-Jan-02-2025_10-55-26.pth'
    # Caser cellphone
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/Caser-Jan-02-2025_10-54-26.pth'
    # FEARec cellphone
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/FEARec-Jan-15-2025_20-53-02.pth'
    # FEARec sports
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/FEARec-Jan-15-2025_20-53-20.pth'
    # FEARec yelp
    # model_pth = '/data/jiarenqi/Baseline/RecBole/saved/FEARec-Jan-15-2025_20-53-21.pth'
    # SASRec ml-1m
    # model_pth = '/data/jiarenqi/DUR/RecBole/saved/SASRec-Apr-04-2025_20-29-01.pth'
    # GRU4Rec ml-1m
    # model_pth = '/data/jiarenqi/DUR/RecBole/saved/GRU4Rec-Apr-10-2025_14-33-02.pth'
    # NARM ml-1m
    # model_pth = '/data/jiarenqi/DUR/RecBole/saved/NARM-Apr-10-2025_18-55-42.pth'
    model_RM = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    model_RM.load_state_dict(torch.load(model_pth,  weights_only=False)['state_dict'])

    # model training
    best_valid_score, best_valid_result = trainer.fit_RM(
        train_data, model_RM, valid_data, saved=saved, show_progress=config["show_progress"]
    )

    # model evaluation
    test_result = trainer.evaluate(
        test_data, load_best_model=saved, show_progress=config["show_progress"]
    )

    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    result = {
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }

    if not config["single_spec"]:
        dist.destroy_process_group()

    if config["local_rank"] == 0 and queue is not None:
        queue.put(result)  # for multiprocessing, e.g., mp.spawn

    return result  # for the single process

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="SASRec", help="name of models")
    parser.add_argument(
        "--dataset", "-d", type=str, default="Amazon_Beauty", help="name of datasets"
    )
    parser.add_argument(
        "--ratio", "-r", type=float, default=0.8, help="ratio of data selection"
    )
    parser.add_argument("--config_files", type=str, default=None, help="config files")
    parser.add_argument(
        "--nproc", type=int, default=1, help="the number of process in this group"
    )
    parser.add_argument(
        "--ip", type=str, default="localhost", help="the ip of master node"
    )
    parser.add_argument(
        "--port", type=str, default="5678", help="the port of master node"
    )
    parser.add_argument(
        "--world_size", type=int, default=-1, help="total number of jobs"
    )
    parser.add_argument(
        "--group_offset",
        type=int,
        default=0,
        help="the global rank offset of this group",
    )

    args, _ = parser.parse_known_args()

    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )

    run(
        args.model,
        args.dataset,
        config_file_list=config_file_list,
        nproc=args.nproc,
        world_size=args.world_size,
        ip=args.ip,
        port=args.port,
        group_offset=args.group_offset,
    )
