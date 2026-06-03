
class Config_EchoNet_Video():
    data_path = "dataset/EchoNet"  # 
    data_subpath = "EchoNet" 
    save_path = "./checkpoints/EchoNet/"
    result_path = "./result/EchoNet/"
    tensorboard_path = "./tensorboard/EchoNet/"
    load_path = save_path + "SAMUS_10081703_24_0.9262574595178807.pth"
    save_path_code = "_"

    workers = 8                         # number of data loading workers (default: 8)
    epochs = 50                       # number of total epochs to run (default: 400)
    batch_size = 16                     # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                        # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train"   # the file name of training set
    val_split = "val"       # the file name of testing set
    test_split = "test"     # the file name of testing set # HMCQU
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 5                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "echonet"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = True
    modelname = "SAM"
    semi = True
class Config_CAMUS_Img_Full():
    data_path = "./dataset/CAMUS"  # 
    data_subpath = "CAMUS_Img_Full" 
    save_path = "./checkpoints/CAMUS_Img_Full/"
    result_path = "./result/CAMUS_Img_Full/"
    tensorboard_path = "./tensorboard/CAMUS_Img_Full/"
    load_path = save_path + "CAMUS.pth"
    save_path_code = "_"

    workers = 8                         # number of data loading workers (default: 8)
    epochs = 100                        # number of total epochs to run (default: 400)
    batch_size = 16                     # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                        # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train"   # the file name of training set
    val_split = "val"       # the file name of testing set
    test_split = "test"     # the file name of testing set # HMCQU
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 10                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "camus"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"
    semi = False
class Config_CAMUS_Img_Semi():
    data_path = "./dataset/CAMUS_old"  # 
    data_subpath = "CAMUS_Img_Semi" 
    save_path = "./checkpoints/CAMUS_Img_Semi/"
    result_path = "./result/CAMUS_Img_Semi/"
    tensorboard_path = "./tensorboard/CAMUS_Img_Semi/"
    load_path = save_path + "CAMUS.pth"
    save_path_code = "_"

    workers = 8                         # number of data loading workers (default: 8)
    epochs = 100                        # number of total epochs to run (default: 400)
    batch_size = 16                     # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                        # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train"   # the file name of training set
    val_split = "val"       # the file name of testing set
    test_split = "test"     # the file name of testing set # HMCQU
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 10                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "camus"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = True
    modelname = "SAM"
    semi = True
class Config_CAMUS_Video_Full():
    data_path = "./dataset/CAMUS"  #
    data_subpath = "CAMUS_Video_Full" 
    save_path = "./checkpoints/CAMUS_Video_Full/"
    result_path = "./result/CAMUS_Video_Full/"
    tensorboard_path = "./tensorboard/CAMUS_Video_Full/"
    load_path = save_path + "CAMUS.pth"
    save_path_code = "_"

    workers = 8                         # number of data loading workers (default: 8)
    epochs = 100                       # number of total epochs to run (default: 400)
    batch_size = 16                     # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                        # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train"   # the file name of training set
    val_split = "val"       # the file name of testing set
    test_split = "test"     # the file name of testing set # HMCQU
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 10                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "camus"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = True
    modelname = "SAM"
    semi = False
class Config_CAMUS_Video_Semi():
    data_path = "/home/zy/coding/sam2/dataset/CAMUS" #
    data_subpath = "CAMUS_Video_Semi" 
    save_path = "./checkpoints/CAMUS_Video_Semi/"
    result_path = "./result/CAMUS_Video_Semi/"
    tensorboard_path = "./tensorboard/CAMUS_Video_Semi/"
    load_path = save_path + "CAMUS.pth"
    save_path_code = "_"

    workers = 8                         # number of data loading workers (default: 8)
    epochs = 100                   # number of total epochs to run (default: 400)
    batch_size = 16                     # batch size (default: 4)
    learning_rate = 1e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                        # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train"   # the file name of training set
    val_split = "val"       # the file name of testing set
    test_split = "test"     # the file name of testing set # HMCQU
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 10                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "camus"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = True
    modelname = "SAM"
    semi = True

# ==================================================================================================
def get_config(task="US30K"):
    if task == "EchoNet":
        return Config_EchoNet_Video()
    elif task == "CAMUS_Img_Full":
        return Config_CAMUS_Img_Full()
    elif task == "CAMUS_Img_Semi":
        return Config_CAMUS_Img_Semi()
    elif task == "CAMUS_Video_Full":
        return Config_CAMUS_Video_Full()
    elif task == "CAMUS_Video_Semi":
        return Config_CAMUS_Video_Semi()
    else:
        assert("We do not have the related dataset, please choose another task.")
