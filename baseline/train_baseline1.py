import torch
from data.single_img_dataset import SingleImgDataLoader
from model.baseline_model import BaselineModel
from opt import opt_parse
from sklearn.mertircs import accuracy_score, recall_score, f1_score, confusion_matrix
import time

def eval_on_metric(pred, label, isTrain=True):
    acc = accuracy_score(pred, label)
    recal = recall_score(pred, label)
    f1 = f1_score(pred, label)
    print("acc: {.4f}, recal: {.4f}, f1_score:{.4f}".format(acc, recal, f1))
    if not isTrain:
        # C_ij => i(label), j(pred)
        print("confusion matrix:\n {}".format(confusion_matrix(pred, label)))
    print('\n')

def flatmap(data):
    return data

def validate(model, dataloader):
    print("Eval on valid set:")
    model.eval()
    total_pred = []
    total_label = []
    for data in dataloader:
        pred = model.run_one_batch(data)
        label = data['label']
        total_pred.append(pred)
        total_label.append(pred)
        eval_on_metric


if __name__ == '__main__':
    opt = opt_parse()
    model = BaselineModel(opt)

    # for eval
    tst_dataloader = SingleImgDataLoader(opt, False)
    if opt.evaluate:
        validate(model, tst_dataloader)
        exit()

    trn_dataloader = SingleImgDataLoader(opt, True)
    step = 0

    best_acc, best_acc_epoch = 0.0, 0
    best_f1, best_f1_epoch = 0.0, 0

    for epoch in range(opt.start_epoch, opt.epochs):
        model.train()
        print("Epoch: {} Start")
        total_pred = []
        total_label = []
        cur_step = 0

        epoch_start = time.time()
        for data in trn_dataloader:
            # forward one batch
            pred = model.run_one_batch(data)
            cur_step += opt.train_batch
            step += opt.train_batch
            # record preds
            label = data['label']
            total_pred.append(pred)
            total_label.append(pred)
            # print current ans
            if cur_step > 200:
                cur_step -= 200
                localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                loss = model.get_current_loss()
                print('{} epoch {} current loss in one batch: {}'.format(localtime, epoch, loss))
        print("Evaluate training on epoch:{}".format(epoch))
        eval_on_metric(model, total_pred, total_labels)
        if epoch % 2 == 0:
            acc, f1 = validate(model, tst_dataloader)
            if best_acc < acc:
                best_acc, best_acc_epoch = acc, epoch
            if best_f1 < f1:
                best_f1, best_f1_epoch = f1, epoch
        epoch_end = time.time()
        time_span = str(datetime.timedelta(seconds=int(epoch_end-epoch_start)))
        print('Using {} to finish epoch'.format(time_span, epoch))
        model.save(epoch)
