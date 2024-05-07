from segment_anything import sam_model_registry, SamPredictor
import os
import torch.nn as nn
import torch
import torch.distributed as dist
import argparse
import os
from torch import optim
from torch.utils.data import DataLoader
from DataLoader import TrainingDataset, stack_dict_batched
from utils import FocalDiceloss_IoULoss, get_logger, generate_point, setting_prompt_none
from metrics import SegMetrics
import time
from tqdm import tqdm
import numpy as np
import datetime
from torch.nn import functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import time
import random
import torch.multiprocessing as mp
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from segment_anything.modeling import Sam, PromptEncoder, MaskDecoder,TwoWayTransformer
from efficientvit.models.efficientvit.efficientvim import image_encoder_efficientvim
from segment_any.modeling.image_encoder_vim import ImageEncoderViM

def _build_mamba_sam(
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 256
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViM(img_size=image_size),

        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam

def _build_efiicientvim_sam(
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = 256
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=image_encoder_efficientvim,

        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam

efficientvim_sam = _build_efiicientvim_sam()

def parse_args():   
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="workdir", help="work dir")
    parser.add_argument("--run_name", type=str, default="mamba-sam-med2d", help="run model name")
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=60, help="train batch size")
    parser.add_argument("--image_size", type=int, default=256, help="image_size")
    parser.add_argument("--mask_num", type=int, default=5, help="get mask number")
    parser.add_argument("--data_path", type=str, default="/mnt/dataset/SAMeD", help="train data path") 
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--resume", type=str, default=None, help="load resume")
    parser.add_argument("--sam_checkpoint", type=str, default=None, help="sam checkpoint")
    parser.add_argument("--iter_point", type=int, default=5, help="point iterations")
    parser.add_argument('--lr_scheduler', type=str, default=None, help='lr scheduler')
    parser.add_argument("--point_list", type=list, default=[1, 3, 5, 9], help="point_list")
    parser.add_argument("--multimask", type=bool, default=True, help="ouput multimask")
    parser.add_argument("--used_amp", type=bool, default=True, help="use amp")

    args = parser.parse_args()
    if args.resume is not None:
        args.sam_checkpoint = None
    return args

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '19158'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4' 
world_size = torch.cuda.device_count()
os.environ['WORLD_SIZE'] = str(world_size)

def init_ddp(local_rank):
    torch.cuda.set_device(local_rank)  
    os.environ['RANK'] = str(local_rank)
    dist.init_process_group(backend='gloo', init_method='env://')

def get_ddp_generator(seed=3407):
    local_rank = dist.get_rank()
    g = torch.Generator()
    g.manual_seed(seed + local_rank)
    return g

def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key=='image' or key=='label':
                device_input[key] = value.float().to(device)
            elif type(value) is list or type(value) is torch.Size:
                 device_input[key] = value
            else:
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input

def prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter = False):
    if  batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None

    if decoder_iter:
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = model.module.prompt_encoder(
                points=points,
                boxes=batched_input.get("boxes", None),
                masks=batched_input.get("mask_inputs", None),
            )
    else:
        sparse_embeddings, dense_embeddings = model.module.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
        )

    low_res_masks, iou_predictions = model.module.mask_decoder(
        image_embeddings = image_embeddings.to(dtype=torch.float16 if args.used_amp else torch.float32),

        image_pe = model.module.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=args.multimask,
    )

    if args.multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i+1, idx])
        low_res_masks = torch.stack(low_res, 0)

    masks = F.interpolate(low_res_masks,(args.image_size, args.image_size), mode="bilinear", align_corners=False,)
    return masks, low_res_masks, iou_predictions

def train_one_epoch(args, model, optimizer, train_loader, epoch, criterion, scaler, local_rank):
    if local_rank==0:
        train_loader = tqdm(train_loader)
    train_losses = []
    train_iter_metrics = [0] * len(args.metrics)

    last_save_time = time.time()
    for batch, batched_input in enumerate(train_loader):
        if batched_input['image'] is None:
            if local_rank == 0:
                print(f"Skipping batch {batch} due to invalid data")
            continue
        try:
            batched_input = stack_dict_batched(batched_input)
            batched_input = to_device(batched_input, args.device)
            
            if random.random() > 0.5:
                batched_input["point_coords"] = None
                flag = "boxes"
            else:
                batched_input["boxes"] = None
                flag = "point"

            for _, value in model.module.image_encoder.named_parameters():
                value.requires_grad = True

            
            with autocast(enabled=args.used_amp):
                labels = batched_input["label"]
                image_embeddings = model.module.image_encoder(batched_input["image"])

                batch, _, _, _ = image_embeddings.shape
                image_embeddings_repeat = []
                for i in range(batch):
                    image_embed = image_embeddings[i]
                    image_embed = image_embed.repeat(args.mask_num, 1, 1, 1)
                    image_embeddings_repeat.append(image_embed)
                image_embeddings = torch.cat(image_embeddings_repeat, dim=0)

                masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter=False)
                loss = criterion(masks, labels, iou_predictions)

            scaler.scale(loss).backward(retain_graph=False)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if int(batch+1) % 50 == 0:
                print(f'Epoch: {epoch+1}, Batch: {batch+1}, first {flag} prompt: {SegMetrics(masks, labels, args.metrics)}')

            point_num = random.choice(args.point_list)
            batched_input = generate_point(masks, labels, low_res_masks, batched_input, point_num)
            batched_input = to_device(batched_input, args.device)
        
            image_embeddings = image_embeddings.detach().clone()
            for n, value in model.named_parameters():
                if "image_encoder" in n:
                    value.requires_grad = False
                else:
                    value.requires_grad = True

            init_mask_num = np.random.randint(1, args.iter_point - 1)
            for iter in range(args.iter_point):
                if iter == init_mask_num or iter == args.iter_point - 1:
                    batched_input = setting_prompt_none(batched_input)

                with autocast(enabled=args.used_amp):
                    masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter=True)
                    loss = criterion(masks, labels, iou_predictions)
                scaler.scale(loss).backward(retain_graph=True)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                if iter != args.iter_point - 1:
                    point_num = random.choice(args.point_list)
                    batched_input = generate_point(masks, labels, low_res_masks, batched_input, point_num)
                    batched_input = to_device(batched_input, args.device)
            
            #     if int(batch+1) % 50 == 0:
            #         if iter == init_mask_num or iter == args.iter_point - 1:
            #             print(f'Epoch: {epoch+1}, Batch: {batch+1}, mask prompt: {SegMetrics(masks, labels, args.metrics)}')
            #         else:
            #             print(f'Epoch: {epoch+1}, Batch: {batch+1}, point {point_num} prompt: { SegMetrics(masks, labels, args.metrics)}')

            # if int(batch+1) % 200 == 0:
            #     print(f"epoch:{epoch+1}, iteration:{batch+1}, loss:{loss.item()}")
            #     save_path = os.path.join(f"{args.work_dir}/models", args.run_name, f"epoch{epoch+1}_batch{batch+1}_sam.pth")
            #     state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            #     torch.save(state, save_path)
            
            current_time = time.time()

            if current_time - last_save_time >= 3600: 
                if local_rank == 0:
                    hours_passed = (current_time - last_save_time) // 3600  
                    formatted_time = datetime.datetime.fromtimestamp(current_time).strftime('%Y%m%d-%H%M%S')  
                    print(f"Saving model at epoch {epoch+1}, batch {batch+1}, time {formatted_time} due to time limit.")

                    save_filename = f"epoch{epoch+1}_batch{batch+1}_time{formatted_time}.pth"
                    save_path = os.path.join(args.work_dir, "models", args.run_name, save_filename)
                    state = {'model': model.module.state_dict(), 'optimizer': optimizer.state_dict()}
                    torch.save(state, save_path)
                last_save_time = current_time  
                
            train_losses.append(loss.item())

            gpu_info = {}
            gpu_info['gpu_name'] = args.device 
            if local_rank==0:
                train_loader.set_postfix(train_loss=loss.item(), gpu_info=gpu_info)
                train_loader.update(1)
            train_batch_metrics = SegMetrics(masks, labels, args.metrics)
            train_iter_metrics = [train_iter_metrics[i] + train_batch_metrics[i] for i in range(len(args.metrics))]

            dist.reduce(torch.tensor(train_losses[-1]).cuda(), 0)
            for i in range(len(train_iter_metrics)):
                dist.reduce(torch.tensor(train_iter_metrics[i]).cuda(), 0)
                train_iter_metrics[i] /= world_size

        except Exception as e:
            if local_rank == 0:
                print(f"Skipping batch {batch} due to an exception: {e}")
            continue

    if local_rank == 0:
        train_loader.close()
    return train_losses, train_iter_metrics

def add_module_prefix(state_dict):
    """Add 'module.' prefix to all parameter keys."""
    return {f'module.{k}': v for k, v in state_dict.items()}


def validate(args, model, val_loader, criterion, local_rank):
    model.eval()
    val_losses = []
    val_iter_metrics = [0] * len(args.metrics)
    try:
        with torch.no_grad():
            for batch, batched_input in enumerate(val_loader):
                if batched_input['image'] is None:
                   continue
                try:
                    batched_input = stack_dict_batched(batched_input)
                    batched_input = to_device(batched_input, args.device)

                    labels = batched_input["label"]
                    image_embeddings = model.module.image_encoder(batched_input["image"])

                    batch, _, _, _ = image_embeddings.shape
                    image_embeddings_repeat = []
                    for i in range(batch):
                        image_embed = image_embeddings[i]
                        image_embed = image_embed.repeat(args.mask_num, 1, 1, 1)
                        image_embeddings_repeat.append(image_embed)
                    image_embeddings = torch.cat(image_embeddings_repeat, dim=0)

                    masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings, decoder_iter=False)
                    loss = criterion(masks, labels, iou_predictions)
                    val_losses.append(loss.item())

                    val_batch_metrics = SegMetrics(masks, labels, args.metrics)
                    val_iter_metrics = [val_iter_metrics[i] + val_batch_metrics[i] for i in range(len(args.metrics))]
                except Exception as e:
                    if local_rank == 0:
                        print(f"Skipping validation batch {batch} due to an exception: {e}")
                    continue

        for i in range(len(val_iter_metrics)):
            dist.reduce(torch.tensor(val_iter_metrics[i]).cuda(), 0)
            val_iter_metrics[i] /= world_size
    except:
        if local_rank == 0:
            print(f"Skipping validation batch {batch} due to an exception: {e}")

    return val_losses, val_iter_metrics


def main(local_rank, args):
    init_ddp(local_rank)
 
    model = _build_mamba_sam().cuda()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = FocalDiceloss_IoULoss().cuda()

    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma = 0.5)
        print('*******Use MultiStepLR')

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location='cuda:{}'.format(local_rank))
        model_state_dict = checkpoint['model']
        if not list(model_state_dict.keys())[0].startswith('module.'):
            model_state_dict = add_module_prefix(model_state_dict)

        model.load_state_dict(model_state_dict)  
        optimizer.load_state_dict(checkpoint['optimizer'])  
        print(f"Loaded weights and optimizer state from {args.resume}")

    if args.used_amp:
        if local_rank==0:
            print("Use mixed precision")
    else:
        print('*******Do not use mixed precision')

    train_dataset = TrainingDataset(args.data_path, image_size=args.image_size, mode='train', point_num=1, mask_num=args.mask_num, requires_name=False)
    train_indices, val_indices = train_test_split(range(len(train_dataset)), test_size=0.05, random_state=42)

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)

    train_sampler = DistributedSampler(train_subset)
    val_sampler = DistributedSampler(val_subset)

    g = get_ddp_generator()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True, generator=g)
    val_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)

    if local_rank==0:
        print('*******Train data:', len(train_dataset))


    loggers = get_logger(os.path.join(args.work_dir, "logs", f"{args.run_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M.log')}"))
   
    metrics_file = os.path.join(args.work_dir, f"{args.run_name}_metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write("Epoch,Train Loss,Train IOU,Train Dice,Val Loss,Val IOU,Val Dice\n")

    best_loss = 1e10
    l = len(train_loader)

    for epoch in range(0, args.epochs):
        scaler = GradScaler()
        model.train()
        train_metrics = {}
        start = time.time()
        os.makedirs(os.path.join(f"{args.work_dir}/models", args.run_name), exist_ok=True)
        train_losses, train_iter_metrics = train_one_epoch(args, model, optimizer, train_loader, epoch, criterion,scaler,local_rank)

        if args.lr_scheduler is not None:
            scheduler.step()

        train_iter_metrics = [metric / l for metric in train_iter_metrics]
        train_metrics = {args.metrics[i]: '{:.4f}'.format(train_iter_metrics[i]) for i in range(len(train_iter_metrics))}

        try:
            val_losses, val_iter_metrics = validate(args, model, val_loader, criterion, local_rank)
            if local_rank == 0:
                average_val_loss = np.mean(val_losses)
                val_metrics = {args.metrics[i]: '{:.4f}'.format(val_iter_metrics[i] / len(val_loader)) for i in range(len(val_iter_metrics))}
                print(f"Validation - Average loss: {average_val_loss:.4f}, Metrics: {val_metrics}")
        except Exception as e:
            if local_rank == 0:
                print(f"An exception occurred during validation: {e}")

        if local_rank==0:
            average_loss = np.mean(train_losses)

            lr = scheduler.get_last_lr()[0] if args.lr_scheduler is not None else args.lr

            loggers.info(f"epoch: {epoch + 1}, lr: {lr}, Train average loss: {average_loss:.4f}, metrics: {train_metrics}")
            with open(metrics_file, 'a') as f:
                f.write(f"{epoch + 1},{average_loss:.4f},{train_metrics['iou']},{train_metrics['dice']},{average_val_loss:.4f},{val_metrics['iou']},{val_metrics['dice']}\n")

        if local_rank == 0:
            if average_loss < best_loss:
                best_loss = average_loss
                save_path = os.path.join(args.work_dir, "models", args.run_name, f"epoch{epoch+1}_sam.pth")
                state = {'model': model.module.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(state, save_path)

        end = time.time()
        if local_rank==0:
            print("Run epoch time: %.2fs" % (end - start))
    dist.destroy_process_group()


if __name__ == '__main__':
    args = parse_args()
    mp.spawn(main, args=(args, ), nprocs=torch.cuda.device_count(), join=True) 
    print('Finished')
