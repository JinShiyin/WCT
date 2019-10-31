# coding=UTF-8
import os
import gc
import argparse
import yaml
import torch
from vgg19_normalized import VGG19_normalized
from vgg19_decoders import *
from libs.Loader import Dataset
from libs.utils import batch_whiten_and_color, autoencoder
from torchvision.utils import save_image
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description='WCT by pytorch')
    parser.add_argument('--content_dir', '-c', type=str, default='/data/jsy/datasets/COCO2017/coco_test',
                        help='dir path of content image')
    parser.add_argument('--style_dir', '-s', type=str, default='/data/jsy/datasets/wikiart/wikiart_test',
                        help='dir path of style image')
    parser.add_argument('--gpu', '-g', type=int, default=2,
                        help='GPU ID,-1: CPU')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--config_path', type=str, default='./configs.yml', help='the configs of the project')
    args = parser.parse_args()

    # load the configs
    file = open(args.config_path)
    config = yaml.safe_load(file)

    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
        print(f'# CUDA:{args.gpu} available: {torch.cuda.get_device_name(args.gpu)}')
    else:
        device = 'cpu'

    # set the model
    layer1 = 'relu1_1'
    layer2 = 'relu2_1'
    layer3 = 'relu3_1'
    layer4 = 'relu4_1'
    layer5 = 'relu5_1'
    encoder = VGG19_normalized()
    decoder1 = VGG19Decoder1()
    decoder2 = VGG19Decoder2()
    decoder3 = VGG19Decoder3()
    decoder4 = VGG19Decoder4()
    decoder5 = VGG19Decoder5()
    encoder.load_state_dict(torch.load('./pre_trained_models/vgg19_normalized.pth.tar'))
    decoder1.load_state_dict(torch.load('./pre_trained_models/vgg19_normalized_decoder1.pth.tar'))
    decoder2.load_state_dict(torch.load('./pre_trained_models/vgg19_normalized_decoder2.pth.tar'))
    decoder3.load_state_dict(torch.load('./pre_trained_models/vgg19_normalized_decoder3.pth.tar'))
    decoder4.load_state_dict(torch.load('./pre_trained_models/vgg19_normalized_decoder4.pth.tar'))
    decoder5.load_state_dict(torch.load('./pre_trained_models/vgg19_normalized_decoder5.pth.tar'))

    if config['type'] == 64:
        encoder = encoder.double()
        decoder1 = decoder1.double()
        decoder2 = decoder2.double()
        decoder3 = decoder3.double()
        decoder4 = decoder4.double()
        decoder5 = decoder5.double()

    encoder = encoder.to(device)
    decoder1 = decoder1.to(device)
    decoder2 = decoder2.to(device)
    decoder3 = decoder3.to(device)
    decoder4 = decoder4.to(device)
    decoder5 = decoder5.to(device)

    dataset = Dataset(args.content_dir, args.style_dir, config)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    for i, (c_tensor, s_tensor, c_names, s_names) in enumerate(loader):
        print(i)
        c_tensor = c_tensor.to(device)
        s_tensor = s_tensor.to(device)

        out5 = autoencoder(encoder, layer5, decoder5, c_tensor, s_tensor)
        out4 = autoencoder(encoder, layer4, decoder4, out5, s_tensor)
        del out5
        gc.collect()
        out3 = autoencoder(encoder, layer3, decoder3, out4, s_tensor)
        del out4
        gc.collect()
        out2 = autoencoder(encoder, layer2, decoder2, out3, s_tensor)
        del out3
        gc.collect()
        out1 = autoencoder(encoder, layer1, decoder1, out2, s_tensor)
        del out2
        gc.collect()

        out = torch.cat([c_tensor, s_tensor, out1], dim=0)
        for j in range(c_tensor.size(0)):
            three = torch.cat([c_tensor[j].unsqueeze(0), s_tensor[j].unsqueeze(0), out1[j].unsqueeze(0)], dim=0)
            save_image(three, f'./result/coco_wikiart_1346_test/three/{c_names[j]}_{s_names[j]}.jpg', nrow=1, padding=0)
            save_image(out1[j].unsqueeze(0), f'./result/coco_wikiart_1346_test/one/{c_names[j]}_{s_names[j]}.jpg', padding=0)
        del out1
        gc.collect()


if __name__ == '__main__':
    main()
