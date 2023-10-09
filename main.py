import argparse
from umaa import *
from datetime import datetime
from utils import *
#from evaluate_utils import *
from data_loader import *
import time
import torchsummary
import torch
import eval_methods

parser = argparse.ArgumentParser()
parser.add_argument('-gpu', '--gpu', dest='gpu', type=int, default=0)
parser.add_argument('-mn', '--mname', dest='mname', default='UMAA') # model name
parser.add_argument('-d', '--dname', dest='dname', default='SWAT') # dataset name
parser.add_argument('-w', '--window', dest='window', type=int, default=12) 
parser.add_argument('-drop', '--dropout', dest='dropout', type=float, default=0.1) # 0.1 if drop rate is 10%
parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=32) 
parser.add_argument('-e', '--max_epoch', dest='max_epoch', type=int, default=10)  


def main(args):

    MODEL_NAME = 'model_'+args.mname+'_'+args.dname+'_W'+str(args.window)+'_E'+str(args.max_epoch)+'_B'+str(args.batch_size)+'_D'+str(args.dropout)
    FILE_NAME = './model/'+MODEL_NAME+'.pth'  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    performance = []  # performance information
    threshold = []  # performance information
	
    # load data
    dataset = get_data_loader(args)	
	
    num_heads = 12
	
    idx = 0
    for data in dataset:

        torch.cuda.empty_cache()
		
        # create model
        if args.dname in ['SWAT']:
            d_size = 51
        elif args.dname in ['WADI']:
            d_size = 123
        elif args.dname in ['MSL']:
            d_size = 55
        elif args.dname in ['SMAP']:
            d_size = 25
        w_size = args.window * d_size  
		
        model = UMAA(d_size, args.window, w_size, num_heads, args.dropout)
        model = to_device(model, device)
	
        idx = idx + 1
		
        # training
        print("= "+str(idx)+" ============ TRAINING ")
        history = training(args.max_epoch, model, data.train_loader, data.train_loader, FILE_NAME)

        # load model
        checkpoint = torch.load(FILE_NAME)
        model.encoder.load_state_dict(checkpoint['encoder'])
        model.decoder1.load_state_dict(checkpoint['decoder1'])
        model.decoder2.load_state_dict(checkpoint['decoder2'])
		
        # test
        results = testing(model, data.test_loader, alpha=0.5, beta=0.5)
        y_pred = np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(), results[-1].flatten().detach().cpu().numpy()])	
        accuracy, thre = eval_methods.bf_search(y_pred, data.labels, verbose = False)
        print("Precision: {}, Recall {}, F1-score: {}".format(accuracy[1], accuracy[2], accuracy[0]))

        performance.append(accuracy)
        threshold.append(thre)

    # total evaluation results
    average_p, average_r, average_f1 = evaluation(performance, verbose=False)
    print("Precision: {}, Recall {}, F1-score: {}".format(average_p, average_r, average_f1))
	

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)

