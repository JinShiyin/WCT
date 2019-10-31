# coding=UTF-8
import torch
import gc


def whiten_and_color(cF, sF):
    device = cF.device
    cFSize = cF.size()
    cF = cF.view(cFSize[0], -1)
    c_mean = torch.mean(cF, 1)  # c x (h x w)
    c_mean = c_mean.unsqueeze(1).expand_as(cF)
    cF = cF - c_mean

    contentConv = torch.mm(cF, cF.t()).div(cFSize[1] - 1) + torch.eye(cFSize[0]).type_as(cF).to(device)
    _, c_e, c_v = torch.svd(contentConv, some=False)

    k_c = cFSize[0]
    for i in range(cFSize[0]):
        if c_e[i] < 0.00001:
            k_c = i
            break

    sFSize = sF.size()
    sF = sF.view(sFSize[0], -1)
    s_mean = torch.mean(sF, 1)
    sF = sF - s_mean.unsqueeze(1).expand_as(sF)
    styleConv = torch.mm(sF, sF.t()).div(sFSize[1] - 1)
    _, s_e, s_v = torch.svd(styleConv, some=False)

    k_s = sFSize[0]
    for i in range(sFSize[0]):
        if s_e[i] < 0.00001:
            k_s = i
            break

    c_d = (c_e[0:k_c]).pow(-0.5)
    step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
    step2 = torch.mm(step1, (c_v[:, 0:k_c].t()))
    whiten_cF = torch.mm(step2, cF)

    s_d = (s_e[0:k_s]).pow(0.5)
    targetFeature = torch.mm(torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(s_d)), (s_v[:, 0:k_s].t())), whiten_cF)
    targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
    targetFeature = targetFeature.view(cFSize[0], cFSize[1], cFSize[2])
    return targetFeature


def batch_whiten_and_color(cf, sf):
    csf = []
    for i in range(cf.size(0)):
        csf.append(whiten_and_color(cf[i], sf[i]).unsqueeze(0))
    csf = torch.cat(csf, dim=0)
    return csf


def autoencoder(encoder, layer, decoder, c, s):
    with torch.no_grad():
        cf = encoder(c, layer)
        sf = encoder(s, layer)
        csf = batch_whiten_and_color(cf, sf)
        del cf
        del sf
        gc.collect()
        out = decoder(csf)
    return out


if __name__ == '__main__':
    cf = torch.rand(2, 256, 64, 64).cuda()
    sf = torch.rand(2, 256, 64, 64).cuda()
    csf = batch_whiten_and_color(cf, sf)
    print(csf.size())
    print(csf.device)
