from typing import Any, Dict
from src.utils.logging import get_logger

logger = get_logger(__name__)

def federated(seed:int, Server: Any, Client: Any, Model: Any, n_workers: int = None, **kwargs) -> Dict:
    
    client_configs = kwargs.pop('clients')
    clients = []
    for id, config in client_configs.items():
        params = kwargs | config
        clients.append(Client(id=id, seed=seed, Model=Model, **params))
    
    if n_workers is None:
        n_workers = len(clients)

    server = Server(seed=seed, Model=Model, clients=clients, n_workers=n_workers, **kwargs)
    results = server.run(**kwargs)
    
    return results

if __name__ == '__main__':
    
    import argparse
    import multiprocessing as mp
    import os
    import pickle
    import time
    from pathlib import Path
    from src.ml import servers, clients, models
    from src.utils.config import load_training_config
    from src.utils.logging import configure_logging

    mp.set_start_method('spawn', force=True)

    EXPERIMENT = 'template_experiment'
    MODEL_TYPE = 'GraphSAGE'

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to models config file.', default=f'experiments/{EXPERIMENT}/config/models.yaml')
    parser.add_argument('--results', type=str, help='Path to results file.', default=None)
    parser.add_argument('--seed', type=int, help='Seed.', default=42)
    parser.add_argument('--n_workers', type=int, help='Number of workers. Default is number of clients.', default=None)
    parser.add_argument('--model_type', type=str, help='Model class.', default=MODEL_TYPE)
    parser.add_argument('--device', type=str, help='Device to use (cpu, cuda:0, etc.)', default=None)
    args = parser.parse_args()

    # Derive experiment name from config path if results not specified
    if args.results is None:
        config_path = Path(args.config)
        experiment_name = config_path.parent.parent.name  # Extract experiment name from config path
        args.results = f'experiments/{experiment_name}/results/federated/{args.model_type}/results.pkl'

    # Configure logging with log file in results directory
    results_dir = os.path.dirname(args.results)
    os.makedirs(results_dir, exist_ok=True)
    log_file = os.path.join(results_dir, 'train.log')
    configure_logging(verbose=True, log_file=log_file)

    t = time.time()

    # Load config with auto-discovery of paths and clients
    kwargs = load_training_config(
        args.config,
        args.model_type,
        setting='federated'
    )
    if args.device is not None:
        kwargs['device'] = args.device
    Server = getattr(servers, kwargs['server_type'])
    Client = getattr(clients, kwargs['client_type'])
    Model = getattr(models, args.model_type)
    results = federated(seed=args.seed, Server=Server, Client=Client, Model=Model, n_workers=args.n_workers, **kwargs)
    
    t = time.time() - t
    
    logger.info('Done')
    logger.info(f'Exec time: {t:.2f}s')
    with open(args.results, 'wb') as f:
        pickle.dump(results, f)
    logger.info(f'Saved results to {args.results}')
