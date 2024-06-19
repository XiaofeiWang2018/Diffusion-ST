import argparse
import yaml
import argparse, time, random
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    sr_create_model_and_diffusion,
    add_dict_to_argparser
)
from guided_diffusion.train_util import TrainLoop


def main():
    # Parse command-line arguments and set up distributed training
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    print(args)
    cur_time = time.strftime('%m%d-%H%M', time.localtime())
    save_dir='logs/x' + str(args.SR_times) + '_' + args.dataset_use + '_g' + str(args.gene_num)
    save_dir =save_dir + '_{}'.format(cur_time)
    logger.configure(dir=save_dir+'/')

    logger.log("creating data loader...")
    # Load the super-resolution dataset
    brain_dataset = load_superres_data(args.data_root, args.dataset_use, status='Train', SR_times=args.SR_times,
                                       gene_num=args.gene_num)
    logger.log("creating model...")
    # Create the super-resolution model and diffusion
    model, diffusion = sr_create_model_and_diffusion(args)

    model.to(dist_util.dev())
    # Create the schedule sampler based on the chosen method
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)




    logger.log("training...")
    # Start the training loop
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=brain_dataset,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        SR_times=args.SR_times,
    ).run_loop()


def load_superres_data(data_root,dataset_use,status,SR_times,gene_num):
    # Load the super-resolution data using the specified directories
    return load_data(data_root=data_root,dataset_use=dataset_use,status=status,SR_times=SR_times,gene_num=gene_num)


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to YAML configuration file")
    args = parser.parse_args()

    # Load the configuration from the YAML file
    with open('config/config_train.yaml', "r") as file:
        config = yaml.safe_load(file)

    # Add the configuration values to the argument parser
    add_dict_to_argparser(parser, config)

    return parser


if __name__ == "__main__":

    main()
