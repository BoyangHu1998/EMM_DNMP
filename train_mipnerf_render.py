from core.config import get_config
from core.trainer_mipnerf_render import MipnerfTrainerRender
from core.train_utils import setup_seed

if __name__ == '__main__':

    config = get_config()

    setup_seed(0)
    trainer = MipnerfTrainerRender(config)
    
    trainer.train()