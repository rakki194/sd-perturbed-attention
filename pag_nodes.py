from functools import partial


BACKEND = None

try:
    from comfy.model_patcher import ModelPatcher
    from comfy.samplers import calc_cond_batch
    from comfy.ldm.modules.attention import optimized_attention
    from .pag_utils import (
        parse_unet_blocks,
        perturbed_attention,
        rescale_guidance,
        seg_attention_wrapper,
        swg_pred_calc,
        snf_guidance,
    )

    try:
        from comfy.model_patcher import set_model_options_patch_replace
    except ImportError:
        from .pag_utils import set_model_options_patch_replace

    BACKEND = "ComfyUI"
except ImportError:
    from pag_utils import (
        parse_unet_blocks,
        set_model_options_patch_replace,
        perturbed_attention,
        rescale_guidance,
        seg_attention_wrapper,
        swg_pred_calc,
        snf_guidance,
    )

    try:
        from ldm_patched.modules.model_patcher import ModelPatcher
        from ldm_patched.modules.samplers import calc_cond_uncond_batch
        from ldm_patched.ldm.modules.attention import optimized_attention

        BACKEND = "reForge"
    except ImportError:
        from backend.patcher.base import ModelPatcher
        from backend.sampling.sampling_function import calc_cond_uncond_batch
        from backend.attention import attention_function as optimized_attention

        BACKEND = "Forge"


class PerturbedAttention:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "scale": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "adaptive_scale": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "round": 0.0001,
                    },
                ),
                "unet_block": (["input", "middle", "output"], {"default": "middle"}),
                "unet_block_id": ("INT", {"default": 0}),
                "sigma_start": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 10000.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "sigma_end": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 10000.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "rescale": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "rescale_mode": (["full", "partial", "snf"], {"default": "full"}),
            },
            "optional": {
                "unet_block_list": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/unet"

    def patch(
        self,
        model: ModelPatcher,
        scale: float = 3.0,
        adaptive_scale: float = 0.0,
        unet_block: str = "middle",
        unet_block_id: int = 0,
        sigma_start: float = -1.0,
        sigma_end: float = -1.0,
        rescale: float = 0.0,
        rescale_mode: str = "full",
        unet_block_list: str = "",
    ):
        m = model.clone()

        sigma_start = float("inf") if sigma_start < 0 else sigma_start
        if unet_block_list:
            blocks = parse_unet_blocks(model, unet_block_list)
        else:
            blocks = [(unet_block, unet_block_id, None)]

        def post_cfg_function(args):
            """CFG+PAG"""
            model = args["model"]
            cond_pred = args["cond_denoised"]
            uncond_pred = args["uncond_denoised"]
            cond = args["cond"]
            cfg_result = args["denoised"]
            sigma = args["sigma"]
            model_options = args["model_options"].copy()
            x = args["input"]

            signal_scale = scale
            if adaptive_scale > 0:
                t = 0
                if hasattr(model, "model_sampling"):
                    t = model.model_sampling.timestep(sigma)[0].item()
                else:
                    ts = model.predictor.timestep(sigma)
                    t = ts[0].item()
                signal_scale -= scale * (adaptive_scale**4) * (1000 - t)
                if signal_scale < 0:
                    signal_scale = 0

            if signal_scale == 0 or not (sigma_end < sigma[0] <= sigma_start):
                return cfg_result

            # Replace Self-attention with PAG
            for block in blocks:
                layer, number, index = block
                model_options = set_model_options_patch_replace(
                    model_options, perturbed_attention, "attn1", layer, number, index
                )

            if BACKEND == "ComfyUI":
                (pag_cond_pred,) = calc_cond_batch(
                    model, [cond], x, sigma, model_options
                )
            if BACKEND in {"Forge", "reForge"}:
                (pag_cond_pred, _) = calc_cond_uncond_batch(
                    model, cond, None, x, sigma, model_options
                )

            pag = (cond_pred - pag_cond_pred) * signal_scale

            if rescale_mode == "snf":
                if uncond_pred.any():
                    return uncond_pred + snf_guidance(cfg_result - uncond_pred, pag)
                return cfg_result + pag

            return cfg_result + rescale_guidance(
                pag, cond_pred, cfg_result, rescale, rescale_mode
            )

        m.set_model_sampler_post_cfg_function(post_cfg_function, rescale_mode == "snf")

        return (m,)


class MultiBlockPerturbedAttention:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "middle_scale": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "output_scale": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "output_cfg_weight": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.01,
                        "round": 0.01,
                    },
                ),
                "adaptive_scale": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "round": 0.0001,
                    },
                ),
                "middle_block_id": ("INT", {"default": 0}),
                "output_block_id": ("INT", {"default": 0}),
                "middle_sigma_start": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 10000.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "middle_sigma_end": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 10000.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "output_sigma_start": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 10000.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "output_sigma_end": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 10000.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "rescale": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "rescale_mode": (["full", "partial", "snf"], {"default": "full"}),
            },
            "optional": {
                "middle_block_list": ("STRING", {"default": ""}),
                "output_block_list": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/unet"

    def patch(
        self,
        model: ModelPatcher,
        middle_scale: float = 3.0,
        output_scale: float = 3.0,
        output_cfg_weight: float = 1.0,
        adaptive_scale: float = 0.0,
        middle_block_id: int = 0,
        output_block_id: int = 0,
        middle_sigma_start: float = -1.0,
        middle_sigma_end: float = -1.0,
        output_sigma_start: float = -1.0,
        output_sigma_end: float = -1.0,
        rescale: float = 0.0,
        rescale_mode: str = "full",
        middle_block_list: str = "",
        output_block_list: str = "",
    ):
        m = model.clone()

        middle_sigma_start = (
            float("inf") if middle_sigma_start < 0 else middle_sigma_start
        )
        output_sigma_start = (
            float("inf") if output_sigma_start < 0 else output_sigma_start
        )

        # Parse middle blocks
        if middle_block_list:
            middle_blocks = parse_unet_blocks(model, middle_block_list)
        else:
            middle_blocks = [("middle", middle_block_id, None)]

        # Parse output blocks
        if output_block_list:
            output_blocks = parse_unet_blocks(model, output_block_list)
        else:
            output_blocks = [("output", output_block_id, None)]

        def post_cfg_function(args):
            """CFG+PAG with separate middle and output blocks"""
            model = args["model"]
            cond_pred = args["cond_denoised"]
            uncond_pred = args["uncond_denoised"]
            cond = args["cond"]
            cfg_result = args["denoised"]
            sigma = args["sigma"]
            model_options = args["model_options"].copy()
            x = args["input"]

            # Calculate adaptive scale if enabled
            current_middle_scale = middle_scale
            current_output_scale = output_scale

            if adaptive_scale > 0:
                t = 0
                if hasattr(model, "model_sampling"):
                    t = model.model_sampling.timestep(sigma)[0].item()
                else:
                    ts = model.predictor.timestep(sigma)
                    t = ts[0].item()

                middle_adjustment = middle_scale * (adaptive_scale**4) * (1000 - t)
                output_adjustment = output_scale * (adaptive_scale**4) * (1000 - t)
                current_middle_scale -= middle_adjustment
                current_output_scale -= output_adjustment

                if current_middle_scale < 0:
                    current_middle_scale = 0
                if current_output_scale < 0:
                    current_output_scale = 0

            final_result = cfg_result

            # Process middle blocks
            if current_middle_scale > 0 and (
                middle_sigma_end < sigma[0] <= middle_sigma_start
            ):
                # Replace Self-attention with PAG for middle blocks
                middle_model_options = model_options.copy()
                for block in middle_blocks:
                    layer, number, index = block
                    middle_model_options = set_model_options_patch_replace(
                        middle_model_options,
                        perturbed_attention,
                        "attn1",
                        layer,
                        number,
                        index,
                    )

                if BACKEND == "ComfyUI":
                    (pag_middle_cond_pred,) = calc_cond_batch(
                        model, [cond], x, sigma, middle_model_options
                    )
                if BACKEND in {"Forge", "reForge"}:
                    (pag_middle_cond_pred, _) = calc_cond_uncond_batch(
                        model, cond, None, x, sigma, middle_model_options
                    )

                middle_pag = (cond_pred - pag_middle_cond_pred) * current_middle_scale

                if rescale_mode == "snf":
                    if uncond_pred.any():
                        final_result = uncond_pred + snf_guidance(
                            cfg_result - uncond_pred, middle_pag
                        )
                    else:
                        final_result = cfg_result + middle_pag
                else:
                    final_result = cfg_result + rescale_guidance(
                        middle_pag, cond_pred, cfg_result, rescale, rescale_mode
                    )

            # Process output blocks
            if current_output_scale > 0 and (
                output_sigma_end < sigma[0] <= output_sigma_start
            ):
                # Replace Self-attention with PAG for output blocks
                output_model_options = model_options.copy()
                for block in output_blocks:
                    layer, number, index = block
                    output_model_options = set_model_options_patch_replace(
                        output_model_options,
                        perturbed_attention,
                        "attn1",
                        layer,
                        number,
                        index,
                    )

                if BACKEND == "ComfyUI":
                    (pag_output_cond_pred,) = calc_cond_batch(
                        model, [cond], x, sigma, output_model_options
                    )
                if BACKEND in {"Forge", "reForge"}:
                    (pag_output_cond_pred, _) = calc_cond_uncond_batch(
                        model, cond, None, x, sigma, output_model_options
                    )

                output_pag = (cond_pred - pag_output_cond_pred) * current_output_scale

                # Apply CFG weight to output blocks guidance
                if output_cfg_weight != 1.0 and uncond_pred.any():
                    # Extract the CFG component (cond - uncond)
                    cfg_component = cond_pred - uncond_pred
                    # Scale the CFG component
                    adjusted_cfg_component = cfg_component * output_cfg_weight
                    # Recalculate guidance with adjusted CFG
                    output_pag = (
                        cond_pred
                        - pag_output_cond_pred
                        + (adjusted_cfg_component - cfg_component)
                    ) * current_output_scale

                if rescale_mode == "snf":
                    if uncond_pred.any():
                        final_result = uncond_pred + snf_guidance(
                            final_result - uncond_pred, output_pag
                        )
                    else:
                        final_result = final_result + output_pag
                else:
                    final_result = final_result + rescale_guidance(
                        output_pag, cond_pred, final_result, rescale, rescale_mode
                    )

            return final_result

        m.set_model_sampler_post_cfg_function(post_cfg_function, rescale_mode == "snf")

        return (m,)


# GROWL: Guidance Rate Optimization with Weighted Logic
class MultiBlockPerturbedAttentionGROWL:
    @classmethod
    def INPUT_TYPES(s):
        # Start with existing MBPA inputs
        types = MultiBlockPerturbedAttention.INPUT_TYPES()
        # Remove adaptive_scale if it exists in the base class inputs
        if "adaptive_scale" in types["required"]:
            del types["required"]["adaptive_scale"]
        # Add GROWL specific inputs
        types["required"].update(
            {
                "positive_prompt": ("STRING", {"multiline": True, "default": ""}),
                "growl_composition_bias": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.05,
                        "round": 0.01,
                    },
                ),
                "growl_detail_bias": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0,
                        "max": 1.0,
                        "step": 0.05,
                        "round": 0.01,
                    },
                ),
            }
        )
        # Optional inputs remain the same
        return types

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/unet"

    def patch(
        self,
        model: ModelPatcher,
        # Include all original MBPA params EXCEPT adaptive_scale
        middle_scale: float = 3.0,
        output_scale: float = 3.0,
        output_cfg_weight: float = 1.0,
        # adaptive_scale removed
        middle_block_id: int = 0,
        output_block_id: int = 0,
        middle_sigma_start: float = -1.0,
        middle_sigma_end: float = -1.0,
        output_sigma_start: float = -1.0,
        output_sigma_end: float = -1.0,
        rescale: float = 0.0,
        rescale_mode: str = "full",
        middle_block_list: str = "",
        output_block_list: str = "",
        # Add GROWL specific params
        positive_prompt: str = "",
        growl_composition_bias: float = 0.0,
        growl_detail_bias: float = 0.0,
    ):
        m = model.clone()

        # --- GROWL Adaptation Logic ---
        prompt_len = len(positive_prompt)
        # Normalize complexity factor (e.g., based on typical prompt lengths, capped at 1.0)
        # Using 150 as a rough estimate for a 'long' prompt needing max adjustment
        complexity_factor = min(1.0, prompt_len / 150.0)

        # Calculate adapted scales based on baseline, bias, and complexity
        # Composition focus (Middle Blocks)
        adapted_middle_scale = (
            middle_scale * (1 + growl_composition_bias) * (1 + complexity_factor * 0.15)
        )
        # Detail focus (Output Blocks)
        adapted_output_scale = (
            output_scale * (1 + growl_detail_bias) * (1 + complexity_factor * 0.30)
        )

        # Clipping to prevent extreme values (e.g., max 2.5x baseline, min 0)
        adapted_middle_scale = max(
            0.0,
            min(adapted_middle_scale, middle_scale * 2.5 if middle_scale > 0 else 1.0),
        )
        adapted_output_scale = max(
            0.0,
            min(adapted_output_scale, output_scale * 2.5 if output_scale > 0 else 1.0),
        )

        print(
            f"[GROWL] Prompt Length: {prompt_len}, Complexity Factor: {complexity_factor:.2f}"
        )
        print(
            f"[GROWL] Middle Scale: {middle_scale:.2f} -> {adapted_middle_scale:.2f} (Bias: {growl_composition_bias:.2f})"
        )
        print(
            f"[GROWL] Output Scale: {output_scale:.2f} -> {adapted_output_scale:.2f} (Bias: {growl_detail_bias:.2f})"
        )
        # --- End GROWL Logic ---

        middle_sigma_start = (
            float("inf") if middle_sigma_start < 0 else middle_sigma_start
        )
        output_sigma_start = (
            float("inf") if output_sigma_start < 0 else output_sigma_start
        )

        if middle_block_list:
            middle_blocks = parse_unet_blocks(model, middle_block_list)
        else:
            middle_blocks = [("middle", middle_block_id, None)]

        if output_block_list:
            output_blocks = parse_unet_blocks(model, output_block_list)
        else:
            output_blocks = [("output", output_block_id, None)]

        def post_cfg_function(args):
            """CFG+PAG (GROWL Adapted)"""
            model = args["model"]
            cond_pred = args["cond_denoised"]
            uncond_pred = args["uncond_denoised"]
            cond = args["cond"]
            cfg_result = args["denoised"]
            sigma = args["sigma"]
            model_options = args["model_options"].copy()
            x = args["input"]

            final_result = cfg_result

            # --- Middle Block Processing (using adapted_middle_scale) ---
            signal_middle_scale = adapted_middle_scale  # Use adapted scale
            # adaptive_scale logic removed

            if signal_middle_scale > 0 and (
                middle_sigma_end < sigma[0] <= middle_sigma_start
            ):
                middle_model_opts = model_options.copy()
                for block in middle_blocks:
                    layer, number, index = block
                    middle_model_opts = set_model_options_patch_replace(
                        middle_model_opts,
                        perturbed_attention,
                        "attn1",
                        layer,
                        number,
                        index,
                    )

                if BACKEND == "ComfyUI":
                    (pag_middle_pred,) = calc_cond_batch(
                        model, [cond], x, sigma, middle_model_opts
                    )
                if BACKEND in {"Forge", "reForge"}:
                    (pag_middle_pred, _) = calc_cond_uncond_batch(
                        model, cond, None, x, sigma, middle_model_opts
                    )

                middle_pag = (cond_pred - pag_middle_pred) * signal_middle_scale

                if rescale_mode == "snf":
                    if uncond_pred.any():
                        final_result = uncond_pred + snf_guidance(
                            cfg_result - uncond_pred, middle_pag
                        )
                    else:
                        final_result = (
                            final_result + middle_pag
                        )  # Add directly if no uncond
                else:
                    final_result = final_result + rescale_guidance(
                        middle_pag, cond_pred, cfg_result, rescale, rescale_mode
                    )

            # --- Output Block Processing (using adapted_output_scale) ---
            signal_output_scale = adapted_output_scale  # Use adapted scale
            # adaptive_scale logic removed

            if signal_output_scale > 0 and (
                output_sigma_end < sigma[0] <= output_sigma_start
            ):
                output_model_opts = model_options.copy()  # Use original options as base
                for block in output_blocks:
                    layer, number, index = block
                    output_model_opts = set_model_options_patch_replace(
                        output_model_opts,
                        perturbed_attention,
                        "attn1",
                        layer,
                        number,
                        index,
                    )

                if BACKEND == "ComfyUI":
                    (pag_output_pred,) = calc_cond_batch(
                        model, [cond], x, sigma, output_model_opts
                    )
                if BACKEND in {"Forge", "reForge"}:
                    (pag_output_pred, _) = calc_cond_uncond_batch(
                        model, cond, None, x, sigma, output_model_opts
                    )

                output_pag = cond_pred - pag_output_pred

                # Apply output_cfg_weight if specified
                if output_cfg_weight != 1.0 and uncond_pred.any():
                    cfg_component = cond_pred - uncond_pred
                    adjusted_cfg = cfg_component * output_cfg_weight
                    output_pag = output_pag + (
                        adjusted_cfg - cfg_component
                    )  # Adjust the delta

                output_pag *= signal_output_scale  # Apply final scale

                if rescale_mode == "snf":
                    # Important: Add output PAG to the result potentially modified by middle PAG
                    if uncond_pred.any():
                        # Apply SNF relative to the current state, not original CFG
                        uncond_base = uncond_pred  # Or should it be derived from final_result? Testing needed.
                        final_result = uncond_base + snf_guidance(
                            final_result - uncond_base, output_pag
                        )
                    else:
                        final_result = final_result + output_pag
                else:
                    final_result = final_result + rescale_guidance(
                        output_pag,
                        cond_pred,
                        final_result,
                        rescale,
                        rescale_mode,  # Rescale based on current final_result
                    )

            return final_result

        m.set_model_sampler_post_cfg_function(post_cfg_function, rescale_mode == "snf")
        return (m,)


class SmoothedEnergyGuidanceAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "scale": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "blur_sigma": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 9999.0,
                        "step": 0.01,
                        "round": 0.001,
                    },
                ),
                "unet_block": (["input", "middle", "output"], {"default": "middle"}),
                "unet_block_id": ("INT", {"default": 0}),
                "sigma_start": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 10000.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "sigma_end": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 10000.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "rescale": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "rescale_mode": (["full", "partial", "snf"], {"default": "full"}),
            },
            "optional": {
                "unet_block_list": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/unet"

    def patch(
        self,
        model: ModelPatcher,
        scale: float = 3.0,
        blur_sigma: float = -1.0,
        unet_block: str = "middle",
        unet_block_id: int = 0,
        sigma_start: float = -1.0,
        sigma_end: float = -1.0,
        rescale: float = 0.0,
        rescale_mode: str = "full",
        unet_block_list: str = "",
    ):
        m = model.clone()

        sigma_start = float("inf") if sigma_start < 0 else sigma_start
        if unet_block_list:
            blocks = parse_unet_blocks(model, unet_block_list)
        else:
            blocks = [(unet_block, unet_block_id, None)]

        def post_cfg_function(args):
            """CFG+SEG"""
            model = args["model"]
            cond_pred = args["cond_denoised"]
            uncond_pred = args["uncond_denoised"]
            cond = args["cond"]
            cfg_result = args["denoised"]
            sigma = args["sigma"]
            model_options = args["model_options"].copy()
            x = args["input"]

            signal_scale = scale

            if signal_scale == 0 or not (sigma_end < sigma[0] <= sigma_start):
                return cfg_result

            seg_attention = seg_attention_wrapper(optimized_attention, blur_sigma)

            # Replace Self-attention with SEG attention
            for block in blocks:
                layer, number, index = block
                model_options = set_model_options_patch_replace(
                    model_options, seg_attention, "attn1", layer, number, index
                )

            if BACKEND == "ComfyUI":
                (seg_cond_pred,) = calc_cond_batch(
                    model, [cond], x, sigma, model_options
                )
            if BACKEND in {"Forge", "reForge"}:
                (seg_cond_pred, _) = calc_cond_uncond_batch(
                    model, cond, None, x, sigma, model_options
                )

            seg = (cond_pred - seg_cond_pred) * signal_scale

            if rescale_mode == "snf":
                if uncond_pred.any():
                    return uncond_pred + snf_guidance(cfg_result - uncond_pred, seg)
                return cfg_result + seg

            return cfg_result + rescale_guidance(
                seg, cond_pred, cfg_result, rescale, rescale_mode
            )

        m.set_model_sampler_post_cfg_function(post_cfg_function, rescale_mode == "snf")

        return (m,)


class SlidingWindowGuidanceAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "scale": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "tile_width": (
                    "INT",
                    {"default": 768, "min": 16, "max": 16384, "step": 8},
                ),
                "tile_height": (
                    "INT",
                    {"default": 768, "min": 16, "max": 16384, "step": 8},
                ),
                "tile_overlap": (
                    "INT",
                    {"default": 256, "min": 16, "max": 16384, "step": 8},
                ),
                "sigma_start": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 10000.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "sigma_end": (
                    "FLOAT",
                    {
                        "default": 5.42,
                        "min": -1.0,
                        "max": 10000.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/unet"

    def patch(
        self,
        model: ModelPatcher,
        scale: float = 5.0,
        tile_width: int = 768,
        tile_height: int = 768,
        tile_overlap: int = 256,
        sigma_start: float = -1.0,
        sigma_end: float = 5.42,
    ):
        m = model.clone()

        sigma_start = float("inf") if sigma_start < 0 else sigma_start
        tile_width, tile_height, tile_overlap = (
            tile_width // 8,
            tile_height // 8,
            tile_overlap // 8,
        )

        def post_cfg_function(args):
            """CFG+SWG"""
            model = args["model"]
            cond_pred = args["cond_denoised"]
            cond = args["cond"]
            cfg_result = args["denoised"]
            sigma = args["sigma"]
            model_options = args["model_options"].copy()
            x = args["input"]

            signal_scale = scale

            if signal_scale == 0 or not (sigma_end < sigma[0] <= sigma_start):
                return cfg_result

            calc_func = None

            if BACKEND == "ComfyUI":
                calc_func = partial(
                    calc_cond_batch,
                    model=model,
                    conds=[cond],
                    timestep=sigma,
                    model_options=model_options,
                )
            if BACKEND in {"Forge", "reForge"}:
                calc_func = partial(
                    calc_cond_uncond_batch,
                    model=model,
                    cond=cond,
                    uncond=None,
                    timestep=sigma,
                    model_options=model_options,
                )

            swg_pred = swg_pred_calc(
                x, tile_width, tile_height, tile_overlap, calc_func
            )
            swg = (cond_pred - swg_pred) * signal_scale

            return cfg_result + swg

        m.set_model_sampler_post_cfg_function(post_cfg_function)

        return (m,)


# Node Mappings
NODE_CLASS_MAPPINGS = {
    "PerturbedAttention": PerturbedAttention,
    "MultiBlockPerturbedAttention": MultiBlockPerturbedAttention,
    "MultiBlockPerturbedAttentionGROWL": MultiBlockPerturbedAttentionGROWL,  # Add GROWL node
    "SmoothedEnergyGuidanceAdvanced": SmoothedEnergyGuidanceAdvanced,
    "SlidingWindowGuidanceAdvanced": SlidingWindowGuidanceAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PerturbedAttention": "Perturbed Attention Guidance",
    "MultiBlockPerturbedAttention": "Multi-Block PAG",
    "MultiBlockPerturbedAttentionGROWL": "Multi-Block PAG (GROWL)",  # Add GROWL display name
    "SmoothedEnergyGuidanceAdvanced": "Smoothed Energy Guidance Adv.",
    "SlidingWindowGuidanceAdvanced": "Sliding Window Guidance Adv.",
}
