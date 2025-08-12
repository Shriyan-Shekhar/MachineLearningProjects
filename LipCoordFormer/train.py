import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from dataset import MyDataset
import numpy as np
import time
from model import LipNet, LipCoordFormer2, LipCoordFormer3, LipCoordFormer4, LipCoordFormer5, LipNet2, LipCoordFormer6
import torch.optim as optim
from tensorboardX import SummaryWriter
import options as opt
from tqdm import tqdm
import logging
import datetime
from torchvision import models

now = datetime.datetime.now()
date_str = now.strftime("%Y%m%d%H%M%S")

logging.basicConfig(filename=f"LipCoordFormer_{opt.model_prefix}_training_log_{date_str}.txt", level=logging.INFO, format="%(asctime)s - %(message)s")

def dataset2dataloader(dataset, num_workers=opt.num_workers, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=opt.pin_memory,
    )


def show_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group["lr"]]
    return np.array(lr).mean()


def ctc_decode(y):
    y = y.argmax(-1)
    return [MyDataset.ctc_arr2txt(y[_], start=1) for _ in range(y.size(0))]


def test(model, net):
    with torch.no_grad():
        dataset = MyDataset(
            opt.video_path,
            opt.anno_path,
            opt.coords_path,
            opt.val_list,
            opt.vid_padding,
            opt.txt_padding,
            "test",
        )

        print("num_test_data:{}".format(len(dataset.videos)))
        model.eval()
        loader = dataset2dataloader(dataset, shuffle=False)
        loss_list = []
        wer = []
        cer = []
        crit = nn.CTCLoss()
        tic = time.time()
        print("RUNNING VALIDATION")
        pbar = tqdm(loader)
        for i_iter, input in enumerate(pbar):
            vid = input.get("vid").cuda(non_blocking=opt.pin_memory)
            txt = input.get("txt").cuda(non_blocking=opt.pin_memory)
            vid_len = input.get("vid_len").cuda(non_blocking=opt.pin_memory)
            txt_len = input.get("txt_len").cuda(non_blocking=opt.pin_memory)
            coord = input.get("coord").cuda(non_blocking=opt.pin_memory)

            y = net(vid, coord)

            loss = (
                crit(
                    y.transpose(0, 1).log_softmax(-1),
                    txt,
                    vid_len.view(-1),
                    txt_len.view(-1),
                )
                .detach()
                .cpu()
                .numpy()
            )
            loss_list.append(loss)
            pred_txt = ctc_decode(y)

            truth_txt = [MyDataset.arr2txt(txt[_].cpu().numpy(), start=1) for _ in range(txt.size(0))]
            wer.extend(MyDataset.wer(pred_txt, truth_txt))
            cer.extend(MyDataset.cer(pred_txt, truth_txt))
            if i_iter % opt.display == 0:
                v = 1.0 * (time.time() - tic) / (i_iter + 1)
                eta = v * (len(loader) - i_iter) / 3600.0

                print("".join(101 * "-"))
                print("{:<50}|{:>50}".format("predict", "truth"))
                print("".join(101 * "-"))
                for predict, truth in list(zip(pred_txt, truth_txt))[:3]:
                    print("{:<50}|{:>50}".format(predict, truth))
                print("".join(101 * "-"))
                print(
                    "test_iter={},eta={},wer={},cer={}".format(
                        i_iter, eta, np.array(wer).mean(), np.array(cer).mean()
                    )
                )
                print("".join(101 * "-"))

        return (np.array(loss_list).mean(), np.array(wer).mean(), np.array(cer).mean())


def train(model, net):
    dataset = MyDataset(
        opt.video_path,
        opt.anno_path,
        opt.coords_path,
        opt.train_list,
        opt.vid_padding,
        opt.txt_padding,
        "train",
    )

    loader = dataset2dataloader(dataset)
    optimizer = optim.Adam(
        model.parameters(), lr=opt.base_lr, weight_decay=0.0, amsgrad=True
    )

    print("num_train_data:{}".format(len(dataset.videos)))
    
    
    crit = nn.CTCLoss()
    tic = time.time()

    train_wer = []
    
    alltime_wer = []
    alltime_cer = []

    print("="*20)
    print(f"Start training on {opt.model_prefix}")
    print("="*20)
    
    for epoch in range(opt.max_epoch):
        print(f"RUNNING EPOCH {epoch}/{opt.max_epoch}")
        pbar = tqdm(loader)

        total_loss = 0.0

        for i_iter, input in enumerate(pbar):
            model.train()
            vid = input.get("vid").cuda(non_blocking=opt.pin_memory)
            txt = input.get("txt").cuda(non_blocking=opt.pin_memory)
            vid_len = input.get("vid_len").cuda(non_blocking=opt.pin_memory)
            txt_len = input.get("txt_len").cuda(non_blocking=opt.pin_memory)
            coord = input.get("coord").cuda(non_blocking=opt.pin_memory)

            optimizer.zero_grad()
            y = net(vid, coord)

            # print(f"y: {y.transpose(0, 1).log_softmax(-1).shape}, txt: {txt.shape}, vid_len: {vid_len.view(-1)}, txt_len: {txt_len.view(-1),}")
            loss = crit(
                y.transpose(0, 1).log_softmax(-1),
                txt,
                vid_len.view(-1),
                txt_len.view(-1),
            )

            total_loss += loss.item()
            
            loss.backward()

            if opt.is_optimize:
                optimizer.step()

            tot_iter = i_iter + epoch * len(loader)

            pred_txt = ctc_decode(y)
            
            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            train_wer.extend(MyDataset.wer(pred_txt, truth_txt))

            if tot_iter % opt.display == 0:
                v = 1.0 * (time.time() - tic) / (tot_iter + 1)
                eta = (len(loader) - i_iter) * v / 3600.0
                print("".join(101 * "-"))
                print("{:<50}|{:>50}".format("predict", "truth"))
                print("".join(101 * "-"))

                for predict, truth in list(zip(pred_txt, truth_txt))[:3]:
                    print("{:<50}|{:>50}".format(predict, truth))
                print("".join(101 * "-"))
                print(
                    "epoch={},tot_iter={},eta={},loss={},train_wer={}".format(
                        epoch, tot_iter, eta, loss, np.array(train_wer).mean()
                    )
                )
                logging.info("epoch={},tot_iter={},eta={},loss={},train_wer={}".format(
                        epoch, tot_iter, eta, loss, np.array(train_wer).mean()
                    ))
                print("".join(101 * "-"))
            '''
            if tot_iter % opt.test_step == 0:
                (loss, wer, cer) = test(model, net)
                print("Testing")
                print(
                    "i_iter={},lr={},loss={},wer={},cer={}".format(
                        tot_iter, show_lr(optimizer), loss, wer, cer
                    )
                )
                # writer.add_scalar("val loss", loss, tot_iter)
                # writer.add_scalar("wer", wer, tot_iter)
                # writer.add_scalar("cer", cer, tot_iter)
                # Get current date and time in format DDMMYYYY_HHMMSS
                now = datetime.datetime.now()
                date_str = now.strftime("%Y%m%d%H%M%S")
                
                savename = "{}_date_{}_loss_{}_wer_{}_cer_{}.pt".format(
                    opt.save_prefix, date_str, loss, wer, cer
                )
                (path, name) = os.path.split(savename)
                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(model.state_dict(), savename)
                if not opt.is_optimize:
                    exit()
                print(f"Saved training checkpoint to {savename}")
            '''
        print("epoch ", epoch, "finished training. Starting testing: ")
        avg_loss = total_loss / len(loader)
        logging.info(f"Epoch [{epoch}/{opt.max_epoch}], Average Training Loss: {avg_loss}")
        (loss, wer, cer) = test(model, net)
        print(
            "lr={},loss={},wer={},cer={}".format(
                show_lr(optimizer), loss, wer, cer
            )
        )
        logging.info(f"Epoch [{epoch}/{opt.max_epoch}] Validation - Validation Loss: {loss}, CER: {cer}, WER: {wer}")
        # writer.add_scalar("val loss", loss, tot_iter)
        # writer.add_scalar("wer", wer, tot_iter)
        # writer.add_scalar("cer", cer, tot_iter)
        #if epoch % 10 == 0 or (epoch + 1) % 10 == 0:
        
        # Save checkpoint if wer or cer perform better than previous
        if epoch == 0 or wer < min(alltime_wer) or cer < min(alltime_cer):
            savename = "LipCoordFormer_{}_epoch_{}_loss_{}_wer_{}_cer_{}.pt".format(
                opt.save_prefix, epoch, loss, wer, cer
            )
            (path, name) = os.path.split(savename)
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(model.state_dict(), savename)
            if not opt.is_optimize:
                exit()
            
            print("Saving model to {}".format(savename))

        # Add validation wer and cer to alltime lists
        alltime_wer.append(wer)
        alltime_cer.append(cer)

if __name__ == "__main__":
    print("Loading options...")
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    writer = SummaryWriter()

    # Load pretrained lipnet, no need for resnet architecture
    
    original_model = LipNet()
    original_model = original_model.cuda()
    # net = nn.DataParallel(original_model).cuda()

    pretrained_weights = "../grid/pretrained_weights/LipNet_unseen_loss_0.44562849402427673_wer_0.1332580699113564_cer_0.06796452465503355.pt"
    
    pretrained_dict = torch.load(pretrained_weights)
    model_dict = original_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}

    for k, param in pretrained_dict.items():
        param.requires_grad = False

    model_dict.update(pretrained_dict)
    original_model.load_state_dict(model_dict)
    
    # model = LipCoordFormer()
    #model = LipCoordFormer2(original_model)
    print("Loading new model...")
    #model = LipCoordFormer3(original_model)
    model = LipCoordFormer4(original_model)
    #model = LipCoordFormer5(original_model)
    #model = LipNet2()
    model = LipCoordFormer6(landmark=True, LSTM=False, transformer=False)
    
    
    
    model = model.cuda()
    net = nn.DataParallel(model).cuda()
    
    '''
    for param in model.feature_extractor.parameters():
        param.requires_grad = False
    print("Frozen original_model layers")
    '''
    
    if hasattr(opt, "weights"):
        print(f"Loading full model weights from {opt.weights}")
        pretrained_dict = torch.load(opt.weights)
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict.keys() and v.size() == model_dict[k].size()
        }

        # freeze the pretrained layers
        # for k, param in pretrained_dict.items():
        #     param.requires_grad = False

        missed_params = [
            k for k, v in model_dict.items() if not k in pretrained_dict.keys()
        ]
        print(
            "loaded params/tot params:{}/{}".format(
                len(pretrained_dict), len(model_dict)
            )
        )
        print("miss matched params:{}".format(missed_params))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    torch.manual_seed(opt.random_seed)
    torch.cuda.manual_seed_all(opt.random_seed)
    torch.backends.cudnn.benchmark = True

    train(model, net)
    