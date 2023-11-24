try:
    import breaching
except ModuleNotFoundError:
    # You only really need this safety net if you want to run these notebooks directly in the examples directory
    # Don't worry about this if you installed the package or moved the notebook to the main directory.
    import os; os.chdir("..")
    import breaching

import torch
import logging, sys
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format='%(message)s')
logger = logging.getLogger()


############# CONFIGURATION OBJECT #############
cfg = breaching.get_config(overrides=["attack=seethroughgradients","case=6_large_batch_cifar"])
          
device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
#device = torch.device('cpu')
torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))
setup

########### ATTACK OPTIONS ###########
#cfg.case.user.user_idx = 0
#cfg.case.user.num_data_points = 1 # How many data points does this user own
#
#cfg.case.user.provide_labels = False
#cfg.attack.label_strategy = "yin" # also works here, as labels are unique
#
## In this paper, there are no public buffers, but users send their batch norm statistic updates in combination 
## with their gradient updates to the server:
#cfg.case.server.provide_public_buffers = False
#cfg.case.user.provide_buffers = True
#
## dataset and model instantiation
#cfg.case.data.shape = [3, 250, 250]
#cfg.case.data.normalize = False
#cfg.case.data.augmentations_val = {}
#cfg.case.data.augmentations_train = {}
#cfg.case.data.name = 'LFWPeople'
#cfg.case.data.partition = 'random'
#cfg.case.data.path = '~/data/imagenet'
#cfg.case.data.size = 13233
#cfg.case.data.classes = 5749
#cfg.case.data.normalize = False

cfg.case.data.partition="random"
cfg.case.user.user_idx = 0
cfg.case.data.name = 'LFWPeople'
cfg.case.data.partition = 'random'
cfg.case.data.path = '~/data/imagenet'
cfg.case.user.num_data_points = 2
cfg.case.data.shape = [3, 250, 250]

cfg.case.model = "vggface2" # also options are resnet50ssl or resnetmoco
cfg.attack.optim.max_iterations = 32000
#cfg.attack.optim.step_size = 0.1
cfg.attack.optim.warmup = 100


#cfg.attack.restarts.num_trials = 5
# In this paper, there are no public buffers, but users send their batch norm statistic updates in combination 
# with their gradient updates to the server:
cfg.case.server.provide_public_buffers = False
cfg.case.user.provide_buffers = True

########### Instantation og parties ###########
user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, setup)
attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack, setup)
breaching.utils.overview(server, user, attacker)

########### Simulation of an attacked FL protocol ##############
server_payload = server.distribute_payload()
shared_data, true_user_data = user.compute_local_updates(server_payload)

user.plot(true_user_data, save_file='/user/gparrella/breaching/my_test/optimization_based/grad_inversion/results/vggface_pre1.png')

########### Reconstruct user data ###########
print("Reconstructing user data...")
# initial data shape example: torch.Size([1, 3, 224, 224])
import torchvision
_default_t = torchvision.transforms.ToTensor()
dataset = torchvision.datasets.LFWPeople(root='~/data/imagenet', split= "test", download=True, transform=_default_t)
# dataset[0][0].unsqueeze(0)
import random
seed1, seed2 = random.randint(0, len(dataset)), random.randint(0, len(dataset))
reconstructed_user_data, stats = attacker.reconstruct(server_payload=[server_payload], shared_data=[shared_data], 
                                                      server_secrets={}, dryrun=cfg.dryrun, initial_data=torch.stack([dataset[seed1][0], dataset[seed2][0]]))

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
user.plot(reconstructed_user_data, save_file='/user/gparrella/breaching/my_test/optimization_based/grad_inversion/results/vggface_post1.png')
