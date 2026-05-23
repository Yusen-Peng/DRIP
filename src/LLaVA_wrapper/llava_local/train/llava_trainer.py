import os
import torch
import torch.nn as nn
import numpy as np
import random

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import Dict, List, Optional
from transformers.utils import is_torch_xla_available


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    # def create_optimizer(self):
    #     """
    #     Setup the optimizer.

    #     We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
    #     Trainer's init through `optimizers`, or subclass and override this method in a subclass.
    #     """
    #     if is_sagemaker_mp_enabled():
    #         return super().create_optimizer()

    #     opt_model = self.model

    #     if self.optimizer is None:
    #         decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
    #         decay_parameters = [name for name in decay_parameters if "bias" not in name]
    #         if self.args.mm_projector_lr is not None:
    #             projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
    #             optimizer_grouped_parameters = [
    #                 {
    #                     "params": [
    #                         p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
    #                     ],
    #                     "weight_decay": self.args.weight_decay,
    #                 },
    #                 {
    #                     "params": [
    #                         p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
    #                     ],
    #                     "weight_decay": 0.0,
    #                 },
    #                 {
    #                     "params": [
    #                         p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
    #                     ],
    #                     "weight_decay": self.args.weight_decay,
    #                     "lr": self.args.mm_projector_lr,
    #                 },
    #                 {
    #                     "params": [
    #                         p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
    #                     ],
    #                     "weight_decay": 0.0,
    #                     "lr": self.args.mm_projector_lr,
    #                 },
    #             ]
    #         else:
    #             optimizer_grouped_parameters = [
    #                 {
    #                     "params": [
    #                         p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
    #                     ],
    #                     "weight_decay": self.args.weight_decay,
    #                 },
    #                 {
    #                     "params": [
    #                         p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
    #                     ],
    #                     "weight_decay": 0.0,
    #                 },
    #             ]

    #         optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

    #         self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    #         if optimizer_cls.__name__ == "Adam8bit":
    #             import bitsandbytes

    #             manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

    #             skipped = 0
    #             for module in opt_model.modules():
    #                 if isinstance(module, nn.Embedding):
    #                     skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
    #                     logger.info(f"skipped {module}: {skipped/2**20}M params")
    #                     manager.register_module_override(module, "weight", {"optim_bits": 32})
    #                     logger.debug(f"bitsandbytes: will optimize {module} in fp32")
    #             logger.info(f"skipped: {skipped/2**20}M params")

    #     return self.optimizer


    # def _load_rng_state(self, checkpoint):
    #     # Load RNG states from `checkpoint`
    #     if checkpoint is None:
    #         return

    #     if self.args.world_size > 1:
    #         process_index = self.args.process_index
    #         rng_file = os.path.join(checkpoint, f"rng_state_{process_index}.pth")
    #         if not os.path.isfile(rng_file):
    #             logger.info(
    #                 f"Didn't find an RNG file for process {process_index}, if you are resuming a training that "
    #                 "wasn't launched in a distributed fashion, reproducibility is not guaranteed."
    #             )
    #             return
    #     else:
    #         rng_file = os.path.join(checkpoint, "rng_state.pth")
    #         if not os.path.isfile(rng_file):
    #             logger.info(
    #                 "Didn't find an RNG file, if you are resuming a training that was launched in a distributed "
    #                 "fashion, reproducibility is not guaranteed."
    #             )
    #             return

    #     checkpoint_rng_state = torch.load(rng_file)
    #     random.setstate(checkpoint_rng_state["python"])
    #     np.random.set_state(checkpoint_rng_state["numpy"])
    #     torch.random.set_rng_state(checkpoint_rng_state["cpu"])
    #     if torch.cuda.is_available():
    #         if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
    #             torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])
    #         else:
    #             try:
    #                 torch.cuda.random.set_rng_state(checkpoint_rng_state["cuda"])
    #             except Exception as e:
    #                 logger.info(
    #                     f"Didn't manage to set back the RNG states of the GPU because of the following error:\n {e}"
    #                     "\nThis won't yield the same results as if the training had not been interrupted."
    #                 )
    #     if is_torch_xla_available():
    #         xm.set_rng_state(checkpoint_rng_state["xla"])
    #     if is_torch_npu_available():
    #         if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
    #             torch.npu.random.set_rng_state_all(checkpoint_rng_state["npu"])
    #         else:
    #             try:
    #                 torch.npu.random.set_rng_state(checkpoint_rng_state["npu"])
    #             except Exception as e:
    #                 logger.info(
    #                     f"Didn't manage to set back the RNG states of the NPU because of the following error:\n {e}"
    #                     "\nThis won't yield the same results as if the training had not been interrupted."
    #                 )
    #     if is_torch_mlu_available():
    #         if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
    #             torch.mlu.random.set_rng_state_all(checkpoint_rng_state["mlu"])
    #         else:
    #             try:
    #                 torch.mlu.random.set_rng_state(checkpoint_rng_state["mlu"])
    #             except Exception as e:
    #                 logger.info(
    #                     f"Didn't manage to set back the RNG states of the MLU because of the following error:\n {e}"
    #                     "\nThis won't yield the same results as if the training had not been interrupted."
    #                 )


    def _load_rng_state(self, checkpoint):
        if checkpoint is None:
            return

        if self.args.world_size <= 1:
            rng_file = os.path.join(checkpoint, "rng_state.pth")
        else:
            rng_file = os.path.join(checkpoint, f"rng_state_{self.args.process_index}.pth")

        if not os.path.isfile(rng_file):
            logger.info("Didn't find an RNG file, so skipping RNG state restore.")
            return

        
        # TRUSTED LOCAL CHECKPOINT: force legacy unpickling for RNG state
        checkpoint_rng_state = torch.load(rng_file, weights_only=False)
        
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])

        if torch.cuda.is_available():
            if self.args.world_size <= 1:
                torch.cuda.random.set_rng_state_all(checkpoint_rng_state["cuda"])
            else:
                torch.cuda.random.set_rng_state(checkpoint_rng_state["cuda"])


    # NOTE: mutual exclusive optimizer setup
    def create_optimizer(self):
        """
        Setup the optimizer with mutually-exclusive param buckets:
        - base (LLM etc.)
        - mm_projector (optional separate LR via --mm_projector_lr)
        - vision_tower (optional separate LR via --vision_tower_lr)
        Buckets are further split into decay / no-decay. No parameter appears in more than one group.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            # 1) decay names (HF default) = all except biases + LayerNorms excluded from weight decay
            decay_names = set(get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS))
            decay_names = {n for n in decay_names if "bias" not in n}

            # 2) projector by name is fine
            projector_names = {n for n, _ in opt_model.named_parameters() if "mm_projector" in n}

            # 3) robust vision_tower membership by OBJECT identity (works under wrappers/sharding)
            inner = getattr(opt_model, "get_model", lambda: opt_model)()
            vt_module = getattr(inner, "vision_tower", None)
            vt_param_ids = set()
            if vt_module is not None:
                for p in vt_module.parameters():
                    vt_param_ids.add(id(p))

            def is_decay(n: str) -> bool:
                return n in decay_names

            def is_projector(n: str) -> bool:
                return n in projector_names

            def is_vt(p) -> bool:
                return id(p) in vt_param_ids

            # 4) exclusive buckets
            base_decay, base_nodecay = [], []
            proj_decay, proj_nodecay = [], []
            vt_decay, vt_nodecay     = [], []

            for name, p in opt_model.named_parameters():
                if not p.requires_grad:
                    continue
                if is_projector(name):
                    (proj_decay if is_decay(name) else proj_nodecay).append(p)
                elif is_vt(p):
                    (vt_decay if is_decay(name) else vt_nodecay).append(p)
                else:
                    (base_decay if is_decay(name) else base_nodecay).append(p)

            # 5) build optimizer parameter groups
            groups = []
            # base
            if base_decay:
                groups.append({"params": base_decay, "weight_decay": self.args.weight_decay})
            if base_nodecay:
                groups.append({"params": base_nodecay, "weight_decay": 0.0})
            # projector (custom LR if provided)
            proj_lr = getattr(self.args, "mm_projector_lr", None)
            if proj_decay:
                groups.append(
                    {"params": proj_decay, "weight_decay": self.args.weight_decay, **({"lr": proj_lr} if proj_lr is not None else {})}
                )
            if proj_nodecay:
                groups.append(
                    {"params": proj_nodecay, "weight_decay": 0.0, **({"lr": proj_lr} if proj_lr is not None else {})}
                )
            # vision tower (custom LR if provided)
            vt_lr = getattr(self.args, "vision_tower_lr", None)  # add this to TrainingArguments if you want a separate LR
            if vt_decay:
                groups.append(
                    {"params": vt_decay, "weight_decay": self.args.weight_decay, **({"lr": vt_lr} if vt_lr is not None else {})}
                )
            if vt_nodecay:
                groups.append(
                    {"params": vt_nodecay, "weight_decay": 0.0, **({"lr": vt_lr} if vt_lr is not None else {})}
                )

            # 6) sanity check: no duplicates / no omissions
            grouped_ids = {id(p) for g in groups for p in g["params"]}
            trainable_ids = {id(p) for p in opt_model.parameters() if p.requires_grad}
            assert grouped_ids == trainable_ids, (
                f"Param grouping mismatch: grouped={len(grouped_ids)} trainable={len(trainable_ids)}"
            )

            # 7) instantiate optimizer
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(groups, **optimizer_kwargs)

            # 8) 8-bit Adam embedding override (unchanged from your original)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes
                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug("bitsandbytes: will optimize %s in fp32", module)
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer
    
    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial, metrics)

        # if getattr(self.args, "tune_mm_mlp_adapter", False):
        # bug fixed: always save the projector (and DRIP if applicable)
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        keys_to_match = ["mm_projector", "vision_resampler"]
        if getattr(self.args, "use_im_start_end", False):
            keys_to_match.extend(["embed_tokens", "embed_in"])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(
            self.model.named_parameters(), keys_to_match
        )

        # save DRIP-specific weights separately
        drip_keys_to_match = ["boundary_predictor", "null_token"]
        drip_weight_to_save = get_mm_adapter_state_maybe_zero_3(
            self.model.named_parameters(), drip_keys_to_match
        )

        if self.args.local_rank in (0, -1):
            torch.save(weight_to_save, os.path.join(output_dir, "mm_projector.bin"))
            # only save if DRIP tensors exist
            if len(drip_weight_to_save) > 0:
                torch.save(drip_weight_to_save, os.path.join(output_dir, "drip.bin"))


    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)

        # Store the latest boundary loss and LM loss for logging purposes
        self._latest_boundary_loss = None
        self._latest_lm_loss = None

        if hasattr(outputs, "boundary_loss") and outputs.boundary_loss is not None:
            self._latest_boundary_loss = outputs.boundary_loss.detach().float().item()

        if hasattr(outputs, "lm_loss") and outputs.lm_loss is not None:
            self._latest_lm_loss = outputs.lm_loss.detach().float().item()

        return (loss, outputs) if return_outputs else loss


    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                # import torch_xla.core.xla_model as xm
                # xm.mark_step()
                pass

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()

            if getattr(self, "_latest_boundary_loss", None) is not None:
                logs["latest_boundary_loss"] = self._latest_boundary_loss
            if getattr(self, "_latest_lm_loss", None) is not None:
                logs["latest_lm_loss"] = self._latest_lm_loss

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

