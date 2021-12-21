import torch


###############################################################################
# Checkpoint utilities
###############################################################################


def latest_path(directory, regex='generator-*.pt'):
    """Retrieve the path to the most recent checkpoint"""
    files = directory.glob(regex)
    files.sort(key=lambda file: int(''.join(filter(str.isdigit, file))))
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
    iteration = checkpoint_dict['iteration']

    print("Loaded checkpoint '{}' (iteration {})" .format(
        checkpoint_path,
        iteration))

    return model, optimizer, iteration


def save(model, optimizer, step, file):
    """Save training checkpoint to disk"""
    print(f'Saving model and optimizer at step {step} to {file}')
    checkpoint = {
        'step': step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, file)
