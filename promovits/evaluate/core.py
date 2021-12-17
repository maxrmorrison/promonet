import promovits


###############################################################################
# Perform evaluation
###############################################################################


def datasets(config, datasets, checkpoint, gpus=None):
    """Evaluate the performance of the model on datasets"""
    # Evaluate on each dataset
    for dataset in datasets:

        # Get adaptation partitions for this dataset
        partitions = promovits.load.partition(dataset)
        train_partitions = sorted(list(
            partition for partition in partitions.keys()
            if 'train_adapt' in partition))
        test_partitions = sorted(list(
            partition for partition in partition.keys()
            if 'test_adapt' in partition))

        # Evaluate on each partition
        iterator = zip(train_partitions, test_partitions)
        for train_partition, test_partition in iterator:

            # Index of this adaptation partition
            index = train_partition.split('-')[-1]

            # Output directory for adaptation artifacts
            adapt_directory = (
                promovits.RUNS_DIR /
                config.stem /
                'adapt' /
                dataset /
                index)

            # Output directory for objective evaluation
            objective_directory = (
                promovits.EVAL_DIR /
                'objective' /
                dataset /
                config.stem /
                index)

            # Output directory for subjective evaluation
            subjective_directory = (
                promovits.EVAL_DIR /
                'subjective' /
                dataset /
                config.stem /
                index)

            dataset(
                dataset,
                train_partition,
                test_partition,
                adapt_directory,
                objective_directory,
                subjective_directory,
                checkpoint,
                gpus)

        # TODO - perform adaptation

        # TODO - save original files to disk

        # TODO - generate evaluation files from all adaptation speakers using
        #        core interface

        # TODO - generate pitch, duration, and loudness modified speech with
        #        scales [.5, .717, 1.414, 2.] using core interface

        # TODO - Extract prosody for all files

        # TODO - Perform objective evaluation and save to disk

        pass
