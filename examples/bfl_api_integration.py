#!/usr/bin/env python3
"""Example: Using Gallium with Black Forest Labs' FLUX.2 API.

This example shows how to integrate gallium with BFL's official API
for experiment tracking and comparison.

Requirements:
    pip install elemental-gallium[grid] requests

Setup:
    export BFL_API_KEY="your-api-key"

BFL API Documentation:
    https://docs.bfl.ml/
"""

import base64
import os
import time
from pathlib import Path
from typing import Optional

import gallium
import gallium.flux as gf

# Uncomment when using the real API:
# import requests


BFL_API_URL = "https://api.bfl.ml/v1"


def generate_with_bfl(
    prompt: str,
    seed: int,
    model: str = gf.FLUX_PRO,
    guidance: float = gf.GUIDANCE_DEFAULT,
    steps: int = gf.STEPS_DEFAULT,
    width: int = 1024,
    height: int = 1024,
    output_dir: str = "outputs",
    api_key: Optional[str] = None,
) -> str:
    """Generate an image with BFL's FLUX.2 API and track with gallium.

    Args:
        prompt: Text prompt for generation.
        seed: Random seed for reproducibility.
        model: FLUX.2 model variant (flux.2-pro, flux.2-max, flux.2-flex).
        guidance: Guidance scale (1.0-20.0).
        steps: Number of inference steps (20-50).
        width: Output width.
        height: Output height.
        output_dir: Directory to save images.
        api_key: BFL API key (defaults to BFL_API_KEY env var).

    Returns:
        Path to the saved image.
    """
    api_key = api_key or os.environ.get("BFL_API_KEY")
    if not api_key:
        raise ValueError("BFL_API_KEY environment variable not set")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # BFL API request (uncomment for real usage)
    # headers = {
    #     "x-key": api_key,  # BFL uses x-key header for auth
    #     "Content-Type": "application/json",
    # }
    #
    # # Map model name to BFL endpoint
    # endpoint_map = {
    #     "flux.2-pro": "flux-pro",
    #     "flux.2-max": "flux-max",
    #     "flux.2-flex": "flux-flex",
    # }
    # endpoint = endpoint_map.get(model, "flux-pro")
    #
    # response = requests.post(
    #     f"{BFL_API_URL}/{endpoint}",
    #     headers=headers,
    #     json={
    #         "prompt": prompt,
    #         "seed": seed,
    #         "guidance_scale": guidance,
    #         "num_inference_steps": steps,
    #         "width": width,
    #         "height": height,
    #     },
    # )
    # response.raise_for_status()
    # result = response.json()
    #
    # # Decode and save image
    # image_data = base64.b64decode(result["image"])
    output_path = f"{output_dir}/{model.replace('.', '_')}_{seed}.png"
    # with open(output_path, "wb") as f:
    #     f.write(image_data)

    duration_ms = int((time.time() - start_time) * 1000)

    # Track with gallium
    exp_id = gallium.log(
        prompt=prompt,
        seed=seed,
        path=output_path,
        model=model,
        width=width,
        height=height,
        duration_ms=duration_ms,
        params={
            "guidance": guidance,
            "steps": steps,
            "provider": "bfl",
            "api_version": "v1",
        },
    )

    print(f"Generated: {output_path} (experiment #{exp_id}, {duration_ms}ms)")
    return output_path


def hackathon_workflow():
    """Complete workflow for a hackathon project.

    This demonstrates a typical workflow:
    1. Generate seed variations to find good compositions
    2. Tune guidance scale on the best seed
    3. Compare model quality
    4. Export results for presentation
    """
    print("\n" + "=" * 60)
    print("BFL FLUX.2 Hackathon Workflow with Gallium")
    print("=" * 60)

    # Step 1: Seed exploration
    print("\n[1/4] Seed Exploration")
    print("-" * 40)

    prompt = "A futuristic Tokyo street at night, neon lights reflecting on wet pavement, cyberpunk aesthetic, highly detailed"

    # Generate 4 seed variations
    seed_params = gf.seed_sweep(prompt, seeds=[42, 123, 456, 789])
    for params in seed_params:
        generate_with_bfl(**{k: v for k, v in params.items() if k != "params"},
                         guidance=params["params"]["guidance"],
                         steps=params["params"]["steps"])

    # Create seed comparison grid
    experiments = gallium.find(prompt__contains="futuristic Tokyo")
    if experiments:
        grid = gallium.grid(
            experiments,
            cols=2,
            max_size=512,
            labels=[f"seed={e.seed}" for e in experiments],
        )
        grid.save("outputs/1_seed_exploration.png")
        print("Saved: outputs/1_seed_exploration.png")

    # Step 2: Guidance tuning (using best seed)
    print("\n[2/4] Guidance Scale Tuning")
    print("-" * 40)

    best_seed = 42  # Assume we picked this from step 1
    guidance_params = gf.guidance_sweep(
        prompt,
        guidance_values=[3.0, 5.0, 7.5, 10.0, 15.0],
        seed=best_seed,
    )

    for params in guidance_params:
        generate_with_bfl(**{k: v for k, v in params.items() if k != "params"},
                         guidance=params["params"]["guidance"],
                         steps=params["params"]["steps"])

    # Create guidance comparison
    experiments = gallium.find(seed=best_seed)
    if experiments:
        grid = gf.sweep_grid(experiments, "guidance", cols=5)
        grid.save("outputs/2_guidance_tuning.png")
        print("Saved: outputs/2_guidance_tuning.png")

    # Step 3: Model comparison
    print("\n[3/4] Model Quality Comparison")
    print("-" * 40)

    best_guidance = 7.5  # Assume we picked this from step 2
    model_params = gf.model_sweep(
        prompt,
        seed=best_seed,
        guidance=best_guidance,
    )

    for params in model_params:
        generate_with_bfl(**{k: v for k, v in params.items() if k != "params"},
                         guidance=params["params"]["guidance"],
                         steps=params["params"]["steps"])

    # Create model comparison matrix
    experiments = gallium.find(seed=best_seed)
    if len(experiments) >= 3:
        matrix = gallium.matrix_grid(experiments, rows="model", cols="seed")
        matrix.save("outputs/3_model_comparison.png")
        print("Saved: outputs/3_model_comparison.png")

    # Step 4: Export for presentation
    print("\n[4/4] Exporting Results")
    print("-" * 40)

    # Export all experiments to JSON
    gallium.export("json", path="outputs/experiments.json")
    print("Saved: outputs/experiments.json")

    # Export to HTML gallery
    gallium.export("html", path="outputs/gallery.html", title="FLUX.2 Hackathon Results")
    print("Saved: outputs/gallery.html")

    # Star the best experiments
    best_experiments = gallium.find(seed=best_seed, model=gf.FLUX_PRO)
    for exp in best_experiments:
        gallium.star(exp.id)
        gallium.annotate(exp.id, "Best configuration found during hackathon")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    total = len(gallium.find())
    starred = len(gallium.find(starred=True))
    print(f"Total experiments: {total}")
    print(f"Starred (best): {starred}")
    print(f"\nAll outputs saved to: outputs/")


def quick_compare(prompt: str, num_seeds: int = 4) -> None:
    """Quick comparison helper for hackathon use.

    Generates N seed variations and creates a comparison grid.

    Args:
        prompt: Text prompt for generation.
        num_seeds: Number of seed variations to generate.
    """
    import random

    seeds = [random.randint(1, 999999) for _ in range(num_seeds)]
    params_list = gf.seed_sweep(prompt, seeds)

    for params in params_list:
        generate_with_bfl(**{k: v for k, v in params.items() if k != "params"},
                         guidance=params["params"]["guidance"],
                         steps=params["params"]["steps"])

    experiments = gallium.recent(num_seeds)
    if experiments:
        grid = gallium.grid(
            experiments,
            cols=2,
            max_size=512,
            labels=[f"seed={e.seed}" for e in experiments],
        )
        # Save with timestamp for easy tracking
        timestamp = int(time.time())
        grid.save(f"outputs/quick_compare_{timestamp}.png")
        print(f"Saved: outputs/quick_compare_{timestamp}.png")


if __name__ == "__main__":
    # Initialize with project database
    gallium.init("bfl_hackathon.db")

    print("BFL FLUX.2 API + Gallium Integration")
    print("=" * 50)
    print("\nNote: Set BFL_API_KEY and uncomment API calls for live generation.")
    print("This example demonstrates the integration pattern.\n")

    # Run the full hackathon workflow
    # hackathon_workflow()

    # Or quick comparison
    # quick_compare("A magical forest with glowing mushrooms", num_seeds=4)

    # Show tracked experiments
    print("\nRecent experiments:")
    for exp in gallium.recent(5):
        print(f"  [{exp.id}] {exp.model}: {exp.prompt[:40]}...")
