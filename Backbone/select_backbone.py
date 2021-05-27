from Backbone.resnet import *


def select_resnet(network):
    param = {'feature_size': 256}
    if network == 'resnet18':
        model = resnet18()
        param['feature_size'] = 256
    elif network == 'resnet34':
        model = resnet34()
        param['feature_size'] = 256
    elif network == 'resnet50':
        model = resnet50()
    elif network == 'resnet101':
        model = resnet101()
    else:
        raise IOError('model type is wrong')

    return model, param
