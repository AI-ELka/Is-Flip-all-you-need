import torch
import os

from modules.base_utils.util import generate_full_path


# def checkpoint_callback(model, opt, epoch, iteration, save_iter, output_dir):
#     '''Saves model and optimizer state dicts at fixed intervals.'''
#     if iteration % save_iter == 0 and iteration != 0:
#         os.makedirs(output_dir, exist_ok=True)
#         checkpoint_path = f'{output_dir}model_{str(epoch)}_{str(iteration)}.pth'
#         opt_path = f'{output_dir}model_{str(epoch)}_{str(iteration)}_opt.pth'
#         torch.save(model.state_dict(), generate_full_path(checkpoint_path))
#         torch.save(opt.state_dict(), generate_full_path(opt_path))

def checkpoint_callback(model, opt, epoch, iteration, save_iter, output_dir):
    '''Saves model and optimizer state dicts at fixed intervals.'''
    if iteration % save_iter == 0 and iteration != 0:
        # Crée le dossier s'il n'existe pas
        os.makedirs(output_dir, exist_ok=True)

        # Chemins complets avec os.path.join
        checkpoint_path = os.path.join(output_dir, f'model_{epoch}_{iteration}.pth')
        opt_path        = os.path.join(output_dir, f'model_{epoch}_{iteration}_opt.pth')

        # Sauvegarde
        torch.save(model.state_dict(), checkpoint_path)
        torch.save(opt.state_dict(), opt_path)
