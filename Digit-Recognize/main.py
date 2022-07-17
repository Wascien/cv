from  DataModel import MinistData
from Net import LeNet
from torch.utils.data import DataLoader
from Train import train_LeNet,precise
import argparse
import torch
def parse_args():
    parse=argparse.ArgumentParser(description='Model args')
    parse.add_argument('-img','--image_path',default='/',type=str,required=True)
    parse.add_argument('-label', '--label_path', default='/', type=str, required=True)
    parse.add_argument('-t_img', '--t_image_path', default='/', type=str, required=True)
    parse.add_argument('-t_label', '--t_label_path', default='/', type=str, required=True)
    parse.add_argument('-B','--batch_size',default=64,type=int)
    parse.add_argument('-E','--epochs', default=10, type=int)
    args=parse.parse_args()

    return args

def main(args):
    img_path,label_path,t_image_path,t_label_path,batch_size,epochs=args.image_path,\
    args.label_path,args.t_image_path,args.t_label_path,args.batch_size,args.epochs

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_data=MinistData(img_path,label_path)
    train_iter=DataLoader(train_data,batch_size=batch_size,shuffle=True)

    net=LeNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    loss_fn=torch.nn.CrossEntropyLoss()

    train_LeNet(train_iter,net,optimizer,None,loss_fn,epochs,batch_size,device)

    torch.save(net.state_dict(),'LeNet.bin')

    test_data=MinistData(t_image_path,t_label_path)
    test_iter=DataLoader(test_data,batch_size=batch_size,shuffle=True)

    precise(test_iter,net,len(test_data),device)


if __name__ == '__main__':
    args=parse_args()
    main(args)











