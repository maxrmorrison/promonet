import torch


###############################################################################
# Checkpoint utilities
###############################################################################


def latest_path(directory, regex='generator-*.pt'):
    """Retrieve the path to the most recent checkpoint"""
    # Retrieve checkpoint filenames
    files = list(directory.glob(regex))

    # If no matching checkpoint files, no training has occurred
    if not files:
        return

    # Retrieve latest checkpoint
    files.sort(key=lambda file: int(''.join(filter(str.isdigit, file.stem))))
    return files[-1]


def load(checkpoint_path, model, optimizer=None):
    """Load model checkpoint from file"""
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

    # Restore model
    model.load_state_dict(checkpoint_dict['model'])

    # Restore optimizer
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])

    # Restore training state
    step = checkpoint_dict['step']

    # Maybe restore loss balancer
    if 'balancer' in checkpoint_dict:
        history = checkpoint_dict['balancer']
    else:
        history = None

    print("Loaded checkpoint '{}' (step {})" .format(checkpoint_path, step))

    return model, optimizer, step, history


def save(model, optimizer, step, file, balancer=None):
    """Save training checkpoint to disk"""
    print(f'Saving model and optimizer at step {step} to {file}')
    checkpoint = {
        'step': step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}
    if balancer is not None:
        checkpoint['balancer'] = balancer.history.detach().cpu()
    torch.save(checkpoint, file)
