try:
    import breaching
except ModuleNotFoundError:
    # You only really need this safety net if you want to run these notebooks directly in the examples directory
    # Don't worry about this if you installed the package or moved the notebook to the main directory.
    import os; os.chdir("..")
    import breaching

import torch
import os
import logging, sys
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format='%(message)s')
logger = logging.getLogger()


############# CONFIGURATION OBJECT #############
cfg = breaching.get_config(overrides=["attack=invertinggradients", "case/server=honest-but-curious",
                                      "case/user=local_updates"])
          
device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#device = torch.device('cpu')
torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))
setup

########### ATTACK OPTIONS ###########
cfg.case.data.partition="random"

cfg.case.user.user_idx = 0

cfg.case.user.num_data_points = 1 # How many data points does this user own


cfg.case.user.provide_labels = False
cfg.attack.label_strategy = "yin" # also works here, as labels are unique

# Total variation regularization scale value
cfg.attack.regularization.total_variation.scale = 5e-1

# dataset and model instantiation
cfg.case.data.name = 'flickr_faces'
cfg.case.model = 'resnet50'
cfg.case.data.partition = 'random'

cfg.case.data.path = '/user/gparrella/data/flickr_images'
cfg.case.data.size = len(os.listdir("/user/gparrella/data/flickr_images"))
cfg.case.data.classes = cfg.case.data.size
# optimizer parameters
cfg.attack.optim.max_iterations = 32000
cfg.attack.optim.step_size = 0.1
cfg.attack.optim.warmup = 100

########### Instantation og parties ###########
user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, setup)
attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack, setup)
breaching.utils.overview(server, user, attacker)

########### Simulation of an attacked FL protocol ##############
server_payload = server.distribute_payload()

shared_data, true_user_data = user.compute_local_updates(server_payload)
save_path = '/user/gparrella/breaching/my_test/optimization_based/inverting_gradients/results/'
user.plot(true_user_data, save_file= os.path.join(save_path,'flickr_original_1.jpg') )

########### Reconstruct user data ###########
print("Reconstructing user data...")
import torchvision
_default_t = torchvision.transforms.ToTensor()
dataset = torchvision.datasets.LFWPeople(root='~/data/imagenet', split= "test", download=True, transform=_default_t)
import random

initial_data = torch.stack([ dataset[seed][0] for seed in [random.randint(0, len(dataset)) for _ in range(0, cfg.case.user.num_data_points)] ])
reconstructed_user_data, stats = attacker.reconstruct(server_payload=[server_payload], shared_data=[shared_data], 
                                                      server_secrets={}, dryrun=cfg.dryrun, 
                                                      initial_data=initial_data)
print("Reconstruction stats:")
# compute evaluation metrics 
try:
    metrics = breaching.analysis.report(reconstructed_user_data, true_user_data, [server_payload], 
                                        server.model, order_batch=True, compute_full_iip=False, 
                                        cfg_case=cfg.case, setup=setup)
except Exception as e:
    print(e)
    metrics = None
print(metrics)
# plot reconstructed data
user.plot(reconstructed_user_data, save_file= os.path.join(save_path,'flickr_reconstructed_1.jpg'))
