import torch


def load_preweights(model, preweights):
    # loading the pretrained weights
    state_dict = {}
    preweights = torch.load(preweights)
    train_parameters = model.state_dict()
    for pname, p in train_parameters.items():
        if pname == 'features.0.weight':
            state_dict[pname] = preweights["features.0.weight"]
            print("loading pretrained weights {}".format(pname))
        elif pname == 'features.0.bias':
            state_dict[pname] = preweights["features.0.bias"]
            print("loading pretrained weights {}".format(pname))
        elif pname == 'features.3.weight':
            state_dict[pname] = preweights["features.3.weight"]
            print("loading pretrained weights {}".format(pname))
        elif pname == 'features.3.bias':
            state_dict[pname] = preweights["features.3.bias"]
            print("loading pretrained weights {}".format(pname))
        elif pname == 'fusion_features.conv3.weight':
            state_dict[pname] = preweights["features.6.weight"]
            print("loading pretrained weights {}".format(pname))
        elif pname == 'fusion_features.conv3.bias':
            state_dict[pname] = preweights["features.6.bias"]
            print("loading pretrained weights {}".format(pname))
        elif pname == 'fusion_features.conv4.weight':
            state_dict[pname] = preweights["features.8.weight"]
            print("loading pretrained weights {}".format(pname))
        elif pname == 'fusion_features.conv4.bias':
            state_dict[pname] = preweights["features.8.bias"]
            print("loading pretrained weights {}".format(pname))
        elif pname == 'fusion_features.conv5.weight':
            state_dict[pname] = preweights["features.10.weight"]
            print("loading pretrained weights {}".format(pname))
        elif pname == 'fusion_features.conv5.bias':
            state_dict[pname] = preweights["features.10.bias"]
            print("loading pretrained weights {}".format(pname))
        else:
            state_dict[pname] = train_parameters[pname]
    return state_dict
