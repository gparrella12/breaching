try:
    import breaching
except ModuleNotFoundError:
    # You only really need this safety net if you want to run these notebooks directly in the examples directory
    # Don't worry about this if you installed the package or moved the notebook to the main directory.
    import os; os.chdir("..")
    import breaching

import torch, os
import logging, sys
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)], format='%(message)s')
logger = logging.getLogger()
import random

for index in range(0, 3):
    print("Trial {}\n".format(index))
    ############# CONFIGURATION OBJECT #############
    cfg = breaching.get_config(overrides=["case=5_small_batch_imagenet", "attack=seethroughgradients"])

            
    device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
    setup = dict(device=device, dtype=getattr(torch, cfg.case.impl.dtype))
    setup

    ########### ATTACK OPTIONS ###########
    cfg.case.data.partition="random"
    cfg.case.user.user_idx = 0

    cfg.case.data.name = 'flickr_faces'
    cfg.case.data.path = '/user/gparrella/data/flickr_images'
    cfg.case.data.size = len(os.listdir("/user/gparrella/data/flickr_images"))
    cfg.case.data.classes = cfg.case.data.size

    cfg.case.user.num_data_points = 16

    cfg.case.model = "vggface2"
    cfg.attack.optim.max_iterations = 320000
    #cfg.attack.optim.step_size = 0.1
    cfg.attack.optim.warmup = 100

    seed = random.randint(0, 100000)
    cfg.seed = seed
    print("Seed: {}".format(cfg.seed))
    # In this paper, there are no public buffers, but users send their batch norm statistic updates in combination 
    cfg.case.server.provide_public_buffers = False
    cfg.case.user.provide_buffers = True

    ########### Instantation og parties ###########
    user, server, model, loss_fn = breaching.cases.construct_case(cfg.case, setup)
    attacker = breaching.attacks.prepare_attack(server.model, server.loss, cfg.attack, setup)
    breaching.utils.overview(server, user, attacker)

    ########### Simulation of an attacked FL protocol ##############
    server_payload = server.distribute_payload()
    shared_data, true_user_data = user.compute_local_updates(server_payload)

    user.plot(true_user_data, save_file='/user/gparrella/breaching/my_test/optimization_based/grad_inversion/results/{}_flickr_{}_pre{}.png'.format(cfg.case.model , cfg.case.user.num_data_points ,index))

    ########### Reconstruct user data ###########
    print("Reconstructing user data...")
    # initial data shape example: torch.Size([1, 3, 224, 224])
    import torchvision
    _default_t = torchvision.transforms.ToTensor()
    dataset = torchvision.datasets.LFWPeople(root='~/data/imagenet', split= "test", download=True, transform=_default_t)
    import random
    # seed1, seed2 = random.randint(0, len(dataset)), random.randint(0, len(dataset))
    initial_data = torch.stack([ dataset[seed][0] for seed in [random.randint(0, len(dataset)) for _ in range(0, cfg.case.user.num_data_points)] ])
    print("initial data len: {}".format(len(initial_data)))
    reconstructed_user_data, stats = attacker.reconstruct(server_payload=[server_payload], shared_data=[shared_data], 
                                                        server_secrets={}, dryrun=cfg.dryrun, initial_data=initial_data)

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
    user.plot(reconstructed_user_data, save_file='/user/gparrella/breaching/my_test/optimization_based/grad_inversion/results/{}_flickr_{}_post{}_1.png'.format(cfg.case.model , cfg.case.user.num_data_points ,index))
    print("========================================================================\n")