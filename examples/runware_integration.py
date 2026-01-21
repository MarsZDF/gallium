#!/usr/bin/env python3
"""Example: Using Gallium with Runware's Generative Media API.

Runware provides a fast, scalable API for image generation with FLUX.2 models.
This example shows how to integrate gallium for experiment tracking.

Requirements:
    pip install elemental-gallium[grid] runware

Setup:
    export RUNWARE_API_KEY="your-api-key"

Runware Documentation:
    https://docs.runware.ai/
"""

import time
from pathlib import Path

import gallium
import gallium.flux as gf

# Uncomment when runware is installed:
# from runware import Runware


def generate_with_runware(
    prompt: str,
    seed: int,
    model: str = gf.FLUX_PRO,
    guidance: float = gf.GUIDANCE_DEFAULT,
    steps: int = gf.STEPS_DEFAULT,
    width: int = 1024,
    height: int = 1024,
    output_dir: str = "outputs",
) -> str:
    """Generate an image with Runware and track with gallium.

    Args:
        prompt: Text prompt for generation.
        seed: Random seed for reproducibility.
        model: FLUX.2 model variant.
        guidance: Guidance scale.
        steps: Number of inference steps.
        width: Output width.
        height: Output height.
        output_dir: Directory to save images.

    Returns:
        Path to the saved image.
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize Runware client
    # runware = Runware(api_key=os.environ["RUNWARE_API_KEY"])

    start_time = time.time()

    # Generate image with Runware
    # result = runware.image_inference(
    #     positive_prompt=prompt,
    #     model=model,
    #     seed=seed,
    #     guidance_scale=guidance,
    #     num_inference_steps=steps,
    #     width=width,
    #     height=height,
    # )

    # Save the image
    # image = result.image
    output_path = f"{output_dir}/{model}_{seed}.png"
    # image.save(output_path)

    duration_ms = int((time.time() - start_time) * 1000)

    # Track with gallium
    gallium.log(
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
            "provider": "runware",
        },
    )

    print(f"Generated: {output_path} ({duration_ms}ms)")
    return output_path


def seed_sweep_example():
    """Example: Compare different seeds with the same prompt."""
    print("\n=== Seed Sweep Example ===\n")

    prompt = "A cyberpunk city at night with neon signs, rain-slicked streets"
    seeds = [42, 123, 456, 789]

    # Generate parameter sets
    params_list = gf.seed_sweep(prompt, seeds)

    # Generate each variation
    paths = []
    for params in params_list:
        path = generate_with_runware(
            prompt=params["prompt"],
            seed=params["seed"],
            model=params["model"],
            guidance=params["params"]["guidance"],
            steps=params["params"]["steps"],
            width=params["width"],
            height=params["height"],
        )
        paths.append(path)

    # Create comparison grid
    experiments = gallium.find(prompt__contains="cyberpunk")
    if experiments:
        grid = gallium.grid(
            experiments,
            cols=2,
            max_size=512,
            labels=[f"seed={e.seed}" for e in experiments],
        )
        grid.save("outputs/seed_comparison.png")
        print("\nSaved comparison grid: outputs/seed_comparison.png")


def guidance_sweep_example():
    """Example: Compare different guidance scales."""
    print("\n=== Guidance Sweep Example ===\n")

    prompt = "Portrait of a wise wizard with a long beard"
    guidance_values = [3.0, 5.0, 7.5, 10.0, 15.0]

    params_list = gf.guidance_sweep(prompt, guidance_values, seed=42)

    for params in params_list:
        g = params["params"]["guidance"]
        generate_with_runware(
            prompt=params["prompt"],
            seed=params["seed"],
            model=params["model"],
            guidance=g,
            steps=params["params"]["steps"],
        )

    # Create labeled grid
    experiments = gallium.find(prompt__contains="wizard")
    if experiments:
        grid = gf.sweep_grid(experiments, "guidance", cols=5)
        grid.save("outputs/guidance_comparison.png")
        print("\nSaved comparison grid: outputs/guidance_comparison.png")


def model_comparison_example():
    """Example: Compare FLUX.2 model variants."""
    print("\n=== Model Comparison Example ===\n")

    prompt = "A serene Japanese garden with cherry blossoms"

    params_list = gf.model_sweep(prompt, seed=42)

    for params in params_list:
        generate_with_runware(
            prompt=params["prompt"],
            seed=params["seed"],
            model=params["model"],
            guidance=params["params"]["guidance"],
            steps=params["params"]["steps"],
        )

    # Use matrix_grid to compare models
    experiments = gallium.find(prompt__contains="Japanese garden")
    if experiments:
        matrix = gallium.matrix_grid(experiments, rows="model", cols="seed")
        matrix.save("outputs/model_comparison.png")
        print("\nSaved comparison grid: outputs/model_comparison.png")


def batch_generation_example():
    """Example: Batch generate with log_many for efficiency."""
    print("\n=== Batch Generation Example ===\n")

    prompts = [
        "A majestic lion in the savanna",
        "A playful dolphin jumping from the ocean",
        "A wise owl perched on an ancient tree",
        "A colorful parrot in a tropical rainforest",
    ]

    # Generate all images
    experiment_data = []
    for i, prompt in enumerate(prompts):
        seed = 42 + i
        # image = runware.generate(...)
        path = f"outputs/animals_{i}.png"
        # image.save(path)

        experiment_data.append({
            "prompt": prompt,
            "seed": seed,
            "path": path,
            "model": gf.FLUX_PRO,
            "width": 1024,
            "height": 1024,
            "params": {"guidance": 7.5, "steps": 28},
        })

    # Batch insert all experiments
    ids = gallium.log_many(experiment_data)
    print(f"Logged {len(ids)} experiments in batch")

    # Create a showcase grid
    experiments = gallium.recent(4)
    if experiments:
        grid = gallium.grid(
            experiments,
            cols=2,
            max_size=512,
            labels=[e.prompt[:30] + "..." for e in experiments],
        )
        grid.save("outputs/animals_grid.png")


if __name__ == "__main__":
    # Initialize gallium with a project-specific database
    gallium.init("runware_experiments.db")

    print("Runware + Gallium Integration Examples")
    print("=" * 50)
    print("\nNote: Uncomment the Runware API calls to run live generation.")
    print("This example shows the integration pattern.\n")

    # Run examples (uncomment runware calls above to actually generate)
    # seed_sweep_example()
    # guidance_sweep_example()
    # model_comparison_example()
    # batch_generation_example()

    # Show what was tracked
    print("\nExperiments logged:")
    for exp in gallium.recent(10):
        print(f"  [{exp.id}] {exp.prompt[:40]}... (seed={exp.seed})")
