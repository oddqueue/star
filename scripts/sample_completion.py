import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from star.utils.distributed import require_single_node


def main() -> None:
    require_single_node()
    # Spawned order workers re-import this entrypoint. Keep heavyweight model
    # and Torch imports behind the main guard so worker processes stay CPU-only.
    from star.sampling import STARSampler
    from star.sampling.config import (
        inference_value,
        load_completion_config_from_cli,
    )
    from star.sampling.runners import run_completion
    from star.sampling.runtime import (
        configure_torch,
        initialize_runtime,
        run_rank_synchronized,
    )
    config = load_completion_config_from_cli()
    configure_torch()
    runtime = initialize_runtime()
    sampler = None
    completed = False
    try:
        def initialize_sampler() -> None:
            nonlocal sampler
            sampler = STARSampler.from_pretrained(
                config,
                device=runtime.device,
                num_order_workers=inference_value(
                    config,
                    "num_order_workers",
                    0,
                ),
                order_seed=inference_value(config, "seed", 42),
            )

        run_rank_synchronized(
            runtime,
            initialize_sampler,
            "STAR sampler initialization failed",
        )
        run_completion(config, runtime, sampler)
        completed = True
    finally:
        try:
            if sampler is not None:
                sampler.close()
        finally:
            runtime.close(graceful=completed)


if __name__ == "__main__":
    main()
