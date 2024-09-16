import argparse
import logging
import pathlib
import pprint
import shutil
import sys
import os

import fairseq.tasks
from torch import nn
from torch.autograd import Variable
import torch.autograd as autograd

import numpy as np
import torch
import torch.utils.data
import tqdm

import dataset
import music_x_transformers
from music_x_transformers import sample
import representation
import utils
import musicbert
from fairseq.data import Dictionary

from transformers import Wav2Vec2Model,BertForSequenceClassification, BertTokenizer

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LAMBDA_GP = 10


@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        choices=("sod", "lmd", "lmd_full", "snd"),
        required=True,
        help="dataset key",
    )
    parser.add_argument(
        "-t", "--train_names", type=pathlib.Path, help="training names"
    )
    parser.add_argument(
        "-v", "--valid_names", type=pathlib.Path, help="validation names"
    )
    parser.add_argument(
        "-i", "--in_dir", type=pathlib.Path, help="input data directory"
    )
    parser.add_argument(
        "-o", "--out_dir", type=pathlib.Path, help="output directory"
    )
    # Data
    parser.add_argument(
        "-bs",
        "--batch_size",
        default=1,
        type=int,
        help="batch size",
    )
    parser.add_argument(
        "--use_csv",
        action="store_true",
        help="whether to save outputs in CSV format (default to NPY format)",
    )
    parser.add_argument(
        "--aug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to use data augmentation",
    )
    # Model
    parser.add_argument(
        "--max_seq_len",
        default=1024,
        type=int,
        help="maximum sequence length",
    )
    parser.add_argument(
        "--max_beat",
        default=256,
        type=int,
        help="maximum number of beats",
    )
    parser.add_argument("--dim", default=512, type=int, help="model dimension")
    parser.add_argument(
        "-l", "--layers", default=6, type=int, help="number of layers"
    )
    parser.add_argument(
        "--heads", default=8, type=int, help="number of attention heads"
    )
    parser.add_argument(
        "--dropout", default=0.2, type=float, help="dropout rate"
    )
    parser.add_argument(
        "--abs_pos_emb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to use absolute positional embedding",
    )
    parser.add_argument(
        "--rel_pos_emb",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="whether to use relative positional embedding",
    )
    # Training
    parser.add_argument(
        "--steps",
        default=200000,
        type=int,
        help="number of steps",
    )
    parser.add_argument(
        "--valid_steps",
        default=1000,
        type=int,
        help="validation frequency",
    )
    parser.add_argument(
        "--early_stopping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether to use early stopping",
    )
    parser.add_argument(
        "-e",
        "--early_stopping_tolerance",
        default=10,
        type=int,
        help="number of extra validation rounds before early stopping",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=0.0007,
        type=float,
        help="learning rate",
    )
    parser.add_argument(
        "-lr_d",
        "--learning_rate_discriminator",
        default=0.00001,
        type=float,
        help="learning rate",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        default=5000,
        type=int,
        help="learning rate warmup steps",
    )
    parser.add_argument(
        "--lr_decay_steps",
        default=100000,
        type=int,
        help="learning rate decay end steps",
    )
    parser.add_argument(
        "--lr_decay_multiplier",
        default=0.1,
        type=float,
        help="learning rate multiplier at the end",
    )
    parser.add_argument(
        "--grad_norm_clip",
        default=2.0,
        type=float,
        help="gradient norm clipping",
    )
    # Sampling
    parser.add_argument(
        "--seq_len", default=1024, type=int, help="sequence length to generate"
    )
    parser.add_argument(
        "--temperature",
        nargs="+",
        default=1.0,
        type=float,
        help="sampling temperature (default: 1.0)",
    )
    parser.add_argument(
        "--filter",
        nargs="+",
        default="top_k",
        type=str,
        help="sampling filter (default: 'top_k')",
    )
    parser.add_argument(
        "--filter_threshold",
        nargs="+",
        default=0.9,
        type=float,
        help="sampling filter threshold (default: 0.9)",
    )
    # Others
    parser.add_argument("-g", "--gpu", type=int, help="gpu number")
    parser.add_argument(
        "-j",
        "--jobs",
        default=4,
        type=int,
        help="number of workers for data loading",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="show warnings only"
    )
    return parser.parse_args(args=args, namespace=namespace)


def get_lr_multiplier(
    step, warmup_steps, decay_end_steps, decay_end_multiplier
):
    """Return the learning rate multiplier with a warmup and decay schedule.

    The learning rate multiplier starts from 0 and linearly increases to 1
    after `warmup_steps`. After that, it linearly decreases to
    `decay_end_multiplier` until `decay_end_steps` is reached.

    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if step > decay_end_steps:
        return decay_end_multiplier
    position = (step - warmup_steps) / (decay_end_steps - warmup_steps)
    return 1 - (1 - decay_end_multiplier) * position

def get_loss(g_model, d_model, batch, seq, mask, device, args):
    criterion = nn.BCELoss()
    encoding = representation.load_encoding(args.in_dir / "encoding.json")
    sos = encoding["type_code_map"]["start-of-song"]
    eos = encoding["type_code_map"]["end-of-song"]

    weight1 = 0.1
    weight2 = 0.9

    BCE_stable = torch.nn.BCEWithLogitsLoss()

    tgt_start = torch.zeros((1, 1, 6), dtype=torch.long, device=device)
    tgt_start[:, 0, 0] = sos

    ground_truth = batch["seq"][0]
    ground_truth = ground_truth.unsqueeze(0)

    generated = g_model.derive(
        tgt_start,
        args.seq_len,
        eos_token=eos,
        temperature=args.temperature,
        filter_logits_fn=args.filter,
        filter_thres=args.filter_threshold,
        monotonicity_dim=("type", "beat"),
    )

    ground_truth = ground_truth.to(device)
    generated = generated.to(device)
    ground_truth = ground_truth.to(torch.int)
    generated = generated.to(torch.int)

    output_real = d_model(ground_truth)

    output_fake = d_model(generated)

    output_fake = torch.clamp(output_fake, min=1e-10, max=1.0)
    output_real = torch.clamp(output_real, min=1e-10, max=1.0)

    ground_truth = ground_truth.to(torch.float)
    loss_d = BCE_stable(output_real - output_fake, ground_truth)

    if torch.isnan(loss_d):
        raise ValueError('Loss is NaN!')

    loss_sub = g_model(seq, mask=mask)

    loss_rs = BCE_stable(output_fake - output_real, ground_truth)

    loss_g = weight1 * loss_rs + weight2 * loss_sub

    return loss_d, loss_g


def main():
    """Main function."""
    # Parse the command-line arguments
    args = parse_args()
    criterion = nn.BCELoss()

    # Set default arguments
    if args.dataset is not None:
        if args.train_names is None:
            args.train_names = pathlib.Path(
                f"data/{args.dataset}/processed/train-names.txt"
            )
        if args.valid_names is None:
            args.valid_names = pathlib.Path(
                f"data/{args.dataset}/processed/valid-names.txt"
            )
        if args.in_dir is None:
            args.in_dir = pathlib.Path(f"data/{args.dataset}/processed/notes/")
        if args.out_dir is None:
            args.out_dir = pathlib.Path(f"exp/test_{args.dataset}")

    # Make sure the output directory exists
    args.out_dir.mkdir(exist_ok=True)
    (args.out_dir / "checkpoints").mkdir(exist_ok=True)

    # Set up the logger
    logging.basicConfig(
        level=logging.ERROR if args.quiet else logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(args.out_dir / "train.log", "w"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Log command called
    logging.info(f"Running command: python {' '.join(sys.argv)}")

    # Log arguments
    logging.info(f"Using arguments:\n{pprint.pformat(vars(args))}")

    # Save command-line arguments
    logging.info(f"Saved arguments to {args.out_dir / 'train-args.json'}")
    utils.save_args(args.out_dir / "train-args.json", args)

    # Get the specified device
    device = torch.device(
        f"cuda:{args.gpu}" if args.gpu is not None else "cpu"
    )
    logging.info(f"Using device: {device}")

    # Load the encoding
    encoding = representation.load_encoding(args.in_dir / "encoding.json")

    # Create the dataset and data loader
    logging.info(f"Creating the data loader...")
    train_dataset = dataset.MusicDataset(
        args.train_names,
        args.in_dir,
        encoding,
        max_seq_len=args.max_seq_len,
        max_beat=args.max_beat,
        use_augmentation=args.aug,
        use_csv=args.use_csv,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        num_workers=args.jobs,
        collate_fn=dataset.MusicDataset.collate,
    )
    valid_dataset = dataset.MusicDataset(
        args.valid_names,
        args.in_dir,
        encoding,
        max_seq_len=args.max_seq_len,
        max_beat=args.max_beat,
        use_csv=args.use_csv,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        args.batch_size,
        num_workers=args.jobs,
        collate_fn=dataset.MusicDataset.collate,
    )

    # Create the model
    logging.info(f"Creating model...")
    model = music_x_transformers.MusicXTransformer(
        dim=args.dim,
        encoding=encoding,
        depth=args.layers,
        heads=args.heads,
        max_seq_len=args.max_seq_len,
        max_beat=args.max_beat,
        rotary_pos_emb=args.rel_pos_emb,
        use_abs_pos_emb=args.abs_pos_emb,
        emb_dropout=args.dropout,
        attn_dropout=args.dropout,
        ff_dropout=args.dropout,
    ).to(device)
    checkpoint_path = 'exp/sod/ape/checkpoints/best_model.pt'
    checkpoint = torch.load(checkpoint_path,map_location=device)
    model.load_state_dict(checkpoint)
    # dis_args = argparse.Namespace()
    # dis_args.tokens_per_sample = 8192
    # task = fairseq.tasks.get_task("masked_lm")
    # filename = "dict.txt"
    # dict = Dictionary.load(filename)
    # model_discriminator = musicbert.MusicBERTModel.build_model(dis_args, dict)
    # model_discriminator = model_discriminator.to(device)
    model_discriminator = music_x_transformers.Discriminator()
    model_discriminator = model_discriminator.to(device)

    # Summarize the model
    n_parameters = sum(p.numel() for p in model.parameters())
    n_trainables = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logging.info(f"Number of parameters: {n_parameters}")
    logging.info(f"Number of trainable parameters: {n_trainables}")

    # Create the optimizer
    optimizer = torch.optim.Adagrad(model.parameters(),
                                 args.learning_rate)
    optimizer_d = torch.optim.Adagrad(model_discriminator.parameters(),
                                   args.learning_rate_discriminator,
                                   weight_decay=0.00001,
                                   eps=1e-5)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr_multiplier(
            step,
            args.lr_warmup_steps,
            args.lr_decay_steps,
            args.lr_decay_multiplier,
        ),
    )

    # Create a file to record losses
    loss_csv = open(args.out_dir / "loss.csv", "w")
    loss_csv.write(
        "step,g_loss,d_loss,valid_loss,valid_g_loss,valid_d_loss,"
        "pitch_loss,duration_loss,instrument_loss\n"
    )

    # Initialize variables
    step = 0
    min_val_loss = float("inf")
    if args.early_stopping:
        count_early_stopping = 0

    # Iterate for the specified number of steps
    train_iterator = iter(train_loader)
    while step < args.steps:

        # Training
        logging.info(f"Training...")
        model.train()
        recent_losses = []

        for batch in (pbar := tqdm.tqdm(range(args.valid_steps), ncols=80)):
            # Get next batch
            try:
                batch = next(train_iterator)
            except StopIteration:
                # Reinitialize dataset iterator
                train_iterator = iter(train_loader)
                batch = next(train_iterator)

            #print(batch["name"])
            # Get input and output pair
            seq = batch["seq"].to(device)
            mask = batch["mask"].to(device)
            if seq.size()[1] != 1024:
                continue

            # Update the model parameters
            optimizer.zero_grad()
            optimizer_d.zero_grad()

            loss_d, loss = get_loss(model,
                                    model_discriminator,
                                    batch,
                                    seq,
                                    mask,
                                    device,
                                    args)
            loss_d.backward(retain_graph=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model_discriminator.parameters(), args.grad_norm_clip
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_norm_clip
            )

            optimizer.step()
            optimizer_d.step()

            scheduler.step()

            # Compute the moving average of the loss
            recent_losses.append(float(loss))
            if len(recent_losses) > 10:
                del recent_losses[0]
            train_loss = np.mean(recent_losses)

            pbar.set_postfix(g_loss=f"{train_loss:8.4f}",d_loss=f"{loss_d:8.4f}")

            step += 1


        # Validation
        logging.info(f"Validating...")
        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_losses = [0] * 6
            count = 0
            for batch in valid_loader:
                # Get input and output pair
                seq = batch["seq"].to(device)
                mask = batch["mask"].to(device)
                if seq.size()[1] != 1024:
                    continue
                val_loss_d, val_loss_g = get_loss(model,
                                                    model_discriminator,
                                                    batch,
                                                    seq,
                                                    mask,
                                                    device,
                                                    args)

                total_loss = val_loss_d + val_loss_g

        val_loss = total_loss
        logging.info(f"Validation loss: {val_loss:.4f}")
        logging.info(f"Generator validation loss: {val_loss_g:.4f}")
        logging.info(f"Discriminator validation loss: {val_loss_d:.4f}")


        # Write losses to file
        loss_csv.write(
            f"{step},{train_loss},{loss_d},{val_loss},{val_loss_d},{val_loss_g}\n"
        )

        # Save the model
        checkpoint_filename = args.out_dir / "checkpoints" / f"model_{step}.pt"
        torch.save(model.state_dict(), checkpoint_filename)
        logging.info(f"Saved the model to: {checkpoint_filename}")

        # Copy the model if it is the best model so far
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            shutil.copyfile(
                checkpoint_filename,
                args.out_dir / "checkpoints" / "best_model.pt",
            )
            # Reset the early stopping counter if we found a better model
            if args.early_stopping:
                count_early_stopping = 0
        elif args.early_stopping:
            # Increment the early stopping counter if no improvement is found
            count_early_stopping += 1

        # Early stopping
        if (
            args.early_stopping
            and count_early_stopping > args.early_stopping_tolerance
        ):
            logging.info(
                "Stopped the training for no improvements in "
                f"{args.early_stopping_tolerance} rounds."
            )
            break

    # Log minimum validation loss
    logging.info(f"Minimum validation loss achieved: {min_val_loss}")

    # Save the optimizer states
    optimizer_filename = args.out_dir / "checkpoints" / f"optimizer_{step}.pt"
    torch.save(optimizer.state_dict(), optimizer_filename)
    logging.info(f"Saved the optimizer state to: {optimizer_filename}")

    # Save the scheduler states
    scheduler_filename = args.out_dir / "checkpoints" / f"scheduler_{step}.pt"
    torch.save(scheduler.state_dict(), scheduler_filename)
    logging.info(f"Saved the scheduler state to: {scheduler_filename}")

    # Close the file
    loss_csv.close()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.getcwd()))
    main()
