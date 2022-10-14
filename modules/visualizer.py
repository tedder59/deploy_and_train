from .build import get_visualizer
import numpy as np

def create_visualizer(dataloader, model, cfg):
    num_samples = cfg.NUM_SAMPLES
    num_total = len(dataloader)
    num_samples = min(num_samples, num_total)
    iters = np.random.choice(np.arange(num_total), num_samples)
    vis = get_visualizer(cfg)
    
    def wrapper(engine, logger, event_name):
        step = engine.state.get_event_attrib_value(event_name)
        if (step % num_total) not in iters:
            return

        batch = engine.state.batch
        output = engine.state.output[0]
        preds = model.predict(output)

        out = vis(batch, preds)
        for name, im in out.items():
            logger.writer.add_image(
                tag=f'{step}_{name}',
                img_tensor=im,
                global_step=step,
                dataformats='HWC'
            )

    return wrapper
