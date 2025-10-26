from unsloth import is_bfloat16_supported
from src.core.cot_training.processing_lib.model_handler import ModelHandler
from src.core.cot_training.processing_lib.dataset_handler import DatasetHandler

import logging
import json
import wandb
import torch
import optuna
import pandas as pd
import optuna.visualization as vis

from typing import Any
from pathlib import Path
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig
from optuna.importance import get_param_importances


logger = logging.getLogger(__name__)


class LLMHyperparameterOptimizer:
    def __init__(
        self,
        dataset_handler_class: type[DatasetHandler],
        dataset_path: str,
        formatted_dataset_dir: Path,
        model_loader_class: type[ModelHandler],
        base_model_name: str,
        chat_template: str,
        max_seq_length: int = 2048,
        use_rslora: bool = True,
        use_loftq: bool = False,
        output_dir: str = "./hpo_results",
        n_trials: int = 20,
        per_device_train_batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int|None = None,
        logging_steps: int = 10,
        eval_steps: float = 0.1,
        epochs: int = 3,
        max_steps_per_trial: int = -1,
        num_cpus: int = 1,
        use_deepspeed: bool = False,
    ):
        self.dataset_handler_class = dataset_handler_class
        self.dataset_path = dataset_path
        self.formatted_dataset_dir = formatted_dataset_dir
        self.model_loader_class = model_loader_class
        self.base_model_name = base_model_name
        self.chat_template = chat_template
        self.max_seq_length = max_seq_length
        self.use_rslora = use_rslora
        self.use_loftq = use_loftq
        self.output_dir = output_dir
        self.n_trials = n_trials
        self.num_cpus = num_cpus

        # Fixed training hyperparameters
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.epochs = epochs
        self.max_steps_per_trial = max_steps_per_trial
        self.use_deepspeed = use_deepspeed

        # Cache for dataset - will be populated on first trial
        self._dataset_dict = None
        self._base_tokenizer = None

    def _prepare_dataset_with_tokenizer(self, tokenizer):
        """Prepare dataset using the provided tokenizer (only once)."""
        if self._dataset_dict is not None:
            logger.info("♻️ Reusing cached dataset")
            return self._dataset_dict

        logger.info("📚 Preparing dataset with tokenizer...")
        dataset_handler = self.dataset_handler_class(
            dataset_path=self.dataset_path,
            formatted_dataset_dir=self.formatted_dataset_dir,
            tokenizer=tokenizer,
            num_cpus=self.num_cpus
        )
        self._dataset_dict = dataset_handler.run_pipeline()
        logger.info("✅ Dataset prepared and cached for reuse")
        return self._dataset_dict

    def _get_base_tokenizer(self):
        """Load base tokenizer once for dataset preparation."""
        if self._base_tokenizer is not None:
            return self._base_tokenizer

        logger.info("🔤 Loading base tokenizer for dataset preparation...")
        temp_loader = self.model_loader_class(
            base_model_name=self.base_model_name,
            chat_template=self.chat_template,
            max_seq_length=self.max_seq_length,
            use_rslora=self.use_rslora,
            use_loftq=self.use_loftq,
        )
        temp_loader._load_base_model()
        _, self._base_tokenizer = temp_loader.obtain_components()

        # Clean up the model, keep only tokenizer
        if hasattr(temp_loader, "base_model"):
            del temp_loader.base_model
        if hasattr(temp_loader, "patched_model"):
            del temp_loader.patched_model
        del temp_loader
        torch.cuda.empty_cache()

        logger.info("✅ Base tokenizer loaded and model cleaned up")
        return self._base_tokenizer

    def create_trainer_with_params(self, trial_params: dict) -> tuple[SFTTrainer, ModelHandler]:
        """Create a new model and trainer with specific hyperparameters using your existing class."""

        model_loader = self.model_loader_class(
            base_model_name=self.base_model_name,
            chat_template=self.chat_template,
            max_seq_length=self.max_seq_length,
            lora_r=trial_params["lora_rank"],
            lora_alpha=(
                trial_params["lora_rank"] * 2
                if not self.use_rslora
                else trial_params["lora_rank"]
            ),
            lora_dropout=trial_params["lora_dropout"],
            use_rslora=self.use_rslora,
            use_loftq=self.use_loftq,
        )

        model_loader._load_base_model()
        model_loader.patch_model() # LoRA patch
        model, tokenizer = model_loader.obtain_components()

        logger.info(f"📊 Model trainable parameters:")
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters() # type: ignore

        is_bf16_supported: bool = is_bfloat16_supported()

        if self.warmup_steps is None:
            # Rule of thumb: 3-10% of total steps
            total_steps = (
                len(self._dataset_dict["train"])
                // (self.per_device_train_batch_size * self.gradient_accumulation_steps)
                * self.epochs
            )
            self.warmup_steps = int(0.05 * total_steps)  # 5% warmup

        sft_config_params: dict[str, Any] = {
            "assistant_only_loss": True,
            "use_liger_kernel": True,
            "output_dir": f"{self.output_dir}/trial_{trial_params['trial_number']}",
            "num_train_epochs": self.epochs,
            "max_steps": self.max_steps_per_trial,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "warmup_steps": self.warmup_steps,
            "learning_rate": trial_params["learning_rate"],
            "weight_decay": trial_params["weight_decay"],
            "max_grad_norm": trial_params.get("max_grad_norm", 1.0),
            "logging_steps": self.logging_steps,
            "eval_strategy": "steps",
            "eval_steps": self.eval_steps,
            "save_strategy": "no",  # Don't save checkpoints during HPO
            "fp16": not is_bf16_supported,
            "bf16": is_bf16_supported,
            "optim": trial_params.get("optimizer", "paged_adamw_8bit"),
            "lr_scheduler_type": trial_params.get("lr_scheduler", "cosine"),
            "max_length": self.max_seq_length,
            # "dataset_text_field":"text",
            "packing": False,
            "report_to": ["wandb"],
            "gradient_checkpointing": True,
        }

        training_args = SFTConfig(**sft_config_params)
        trainer = SFTTrainer(
            model=model, # type: ignore
            processing_class=tokenizer,
            args=training_args,
            train_dataset=self._dataset_dict["train"],
            eval_dataset=self._dataset_dict["validation"],
        )

        return trainer, model_loader

    def objective(self, trial) -> float:
        """Optuna objective function."""

        # Initialize W&B for this trial
        wandb.init(
            project="llm-hpo",
            name=f"trial-{trial.number}",
            reinit=True,
            config={
                "trial_number": trial.number,
                "base_model": self.base_model_name,
                "use_rslora": self.use_rslora,
                "use_loftq": self.use_loftq,
            },
            tags=[self.base_model_name, "rslora" if self.use_rslora else "lora"],
        )

        # Sample hyperparameters
        trial_params = {
            "trial_number": trial.number,
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
            "lora_rank": trial.suggest_categorical("lora_rank", [8, 16, 32, 64]),
            "lora_dropout": trial.suggest_float("lora_dropout", 0.0, 0.2),
            "optimizer": trial.suggest_categorical(
                "optimizer", ["adamw_torch", "adamw_8bit", "paged_adamw_8bit"]
            ),
            "lr_scheduler": trial.suggest_categorical(
                "lr_scheduler", ["cosine", "linear", "constant"]
            ),
            "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 1.5),
        }

        logger.info(f"🔍 Trial {trial.number} with params: {trial_params}")
        wandb.config.update(trial_params)

        trainer = None
        model_loader = None

        try:
            trainer, model_loader = self.create_trainer_with_params(trial_params)
            eval_losses = []

            if self.max_steps_per_trial > 0:
                trainer.train()
                eval_metrics = trainer.evaluate()
                eval_loss = eval_metrics["eval_loss"]
                eval_losses.append(eval_loss)

                wandb.log(
                    {
                        "eval_loss": eval_loss,
                        "train_loss": eval_metrics.get("train_loss", None),
                    }
                )

                final_loss = eval_loss
            else:
                # Train epoch by epoch for pruning
                for epoch in range(int(trainer.args.num_train_epochs)):
                    trainer.train(resume_from_checkpoint=False)

                    # Evaluate
                    eval_metrics = trainer.evaluate()
                    eval_loss = eval_metrics["eval_loss"]
                    eval_losses.append(eval_loss)

                    # Log to W&B
                    wandb.log(
                        {
                            "epoch": epoch,
                            "eval_loss": eval_loss,
                            "train_loss": eval_metrics.get("train_loss", None),
                        }
                    )

                    trial.report(eval_loss, step=epoch)
                    # Check if trial should be pruned
                    if trial.should_prune():
                        logger.info(f"⚠️ Trial {trial.number} pruned at epoch {epoch}")
                        wandb.log({"status": "PRUNED", "pruned_at_epoch": epoch})
                        raise optuna.TrialPruned()

                final_loss = eval_losses[-1]

            logger.info(
                f"✅ Trial {trial.number} completed with loss: {final_loss:.4f}"
            )
            wandb.log({"status": "COMPLETED", "final_eval_loss": final_loss})

            return final_loss

        except optuna.TrialPruned:
            wandb.finish(exit_code=0)
            raise
        except Exception as e:
            logger.error(f"❌ Trial {trial.number} failed: {str(e)}")
            wandb.log({"status": "FAILED", "error": str(e)})
            wandb.finish(exit_code=1)
            raise
        finally:
            # Clean up GPU memory
            if trainer is not None:
                del trainer
            if model_loader is not None:
                if hasattr(model_loader, "base_model"):
                    del model_loader.base_model
                if hasattr(model_loader, "patched_model"):
                    del model_loader.patched_model
                del model_loader
            torch.cuda.empty_cache()
            if wandb.run is not None:
                wandb.finish()

    def _generate_analysis_visualizations(self, study):
        """Generate comprehensive analysis visualizations using Optuna and W&B.

        Optuna Visualizations (HTML + PNG):
          - Optimization History: Shows how loss improves over trials
          - Parameter Importances: Bar chart of parameter impact
          - Parallel Coordinates: Multi-dimensional view of all parameters vs loss
          - Intermediate Values: Pruning visualization across epochs
          - Contour Plots: 2D heatmaps for parameter pairs
          - Slice Plots: Marginal effect of each parameter

        W&B Interactive Plots:
          - Individual Scatter Plots: Each parameter vs eval loss
          - Parameter Importance Bar Chart: Interactive version
          - Optimization Progress: Trial-by-trial improvement
          - Trials Data Table: All trial results in one place
          - Static PNG uploads: For easy thesis integration
        """

        logger.info("📊 Generating comprehensive analysis visualizations...")

        # Extract completed trials data
        completed_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]

        if len(completed_trials) < 2:
            logger.warning("⚠️ Not enough completed trials for comprehensive analysis")
            return

        # Create DataFrame from trials
        trials_data = []
        for trial in completed_trials:
            trial_dict = {
                "trial_number": trial.number,
                "eval_loss": trial.value,
                **trial.params,
            }
            trials_data.append(trial_dict)

        df = pd.DataFrame(trials_data)

        # === OPTUNA VISUALIZATIONS (HTML) ===
        try:
            # 1. Optimization history
            fig = vis.plot_optimization_history(study)
            fig.write_html(f"{self.output_dir}/optimization_history.html")

            # 2. Parameter importances
            fig = vis.plot_param_importances(study)
            fig.write_html(f"{self.output_dir}/param_importances.html")

            # 3. Parallel coordinate plot
            fig = vis.plot_parallel_coordinate(study)
            fig.write_html(f"{self.output_dir}/parallel_coordinate.html")

            # 4. Intermediate values (pruning visualization)
            fig = vis.plot_intermediate_values(study)
            fig.write_html(f"{self.output_dir}/intermediate_values.html")

            # 5. Contour plots for parameter pairs
            fig = vis.plot_contour(study)
            fig.write_html(f"{self.output_dir}/contour_plots.html")

            # 6. Slice plot (marginal effects)
            fig = vis.plot_slice(study)
            fig.write_html(f"{self.output_dir}/slice_plots.html")

            logger.info(f"✅ Optuna HTML visualizations saved to {self.output_dir}/")
        except Exception as e:
            logger.warning(f"⚠️ Could not generate Optuna HTML visualizations: {e}")

        # === OPTUNA VISUALIZATIONS (PNG for thesis) ===
        try:
            # Requires kaleido: pip install kaleido
            fig = vis.plot_optimization_history(study)
            fig.write_image(
                f"{self.output_dir}/optimization_history.png", width=1200, height=600
            )

            fig = vis.plot_param_importances(study)
            fig.write_image(
                f"{self.output_dir}/param_importances.png", width=1200, height=600
            )

            fig = vis.plot_parallel_coordinate(study)
            fig.write_image(
                f"{self.output_dir}/parallel_coordinates.png", width=1400, height=700
            )

            fig = vis.plot_intermediate_values(study)
            fig.write_image(
                f"{self.output_dir}/intermediate_values.png", width=1200, height=600
            )

            logger.info(f"✅ Optuna PNG visualizations saved to {self.output_dir}/")
        except Exception as e:
            logger.warning(
                f"⚠️ Could not generate PNG visualizations (install kaleido): {e}"
            )

        # === PARAMETER IMPORTANCE ANALYSIS ===
        try:
            importances = get_param_importances(study)

            logger.info("\n" + "=" * 50)
            logger.info("📈 PARAMETER IMPORTANCE ANALYSIS")
            logger.info("=" * 50)
            for param, importance in sorted(
                importances.items(), key=lambda x: x[1], reverse=True
            ):
                logger.info(
                    f"  {param:25s}: {importance:6.3f} ({importance*100:5.1f}%)"
                )
            logger.info("=" * 50 + "\n")

            # Save importance to JSON
            with open(f"{self.output_dir}/param_importances.json", "w") as f:
                json.dump(importances, f, indent=4)

            # Create importance DataFrame
            imp_df = pd.DataFrame(
                {
                    "hyperparameter": list(importances.keys()),
                    "importance": list(importances.values()),
                }
            ).sort_values("importance", ascending=False)

            imp_df.to_csv(f"{self.output_dir}/param_importances.csv", index=False)

        except Exception as e:
            logger.warning(f"⚠️ Could not compute parameter importance: {e}")
            importances = {}
            imp_df = None

        # === W&B VISUALIZATIONS ===
        try:
            wandb.init(
                project="llm-hpo-analysis",
                name=f"hpo-analysis-{self.base_model_name.replace('/', '-')}",
                reinit=True,
                config={
                    "n_trials": len(study.trials),
                    "n_completed": len(completed_trials),
                    "best_value": study.best_value,
                },
            )

            # 1. Trials table
            trials_table = wandb.Table(dataframe=df)
            wandb.log({"trials_data": trials_table})

            # 2. Individual parameter impact scatter plots
            for param in df.columns:
                if param not in ["trial_number", "eval_loss"]:
                    # Check if parameter is numeric
                    if pd.api.types.is_numeric_dtype(df[param]):
                        scatter_data = [
                            [x, y] for x, y in zip(df[param], df["eval_loss"])
                        ]
                        table = wandb.Table(
                            data=scatter_data, columns=[param, "eval_loss"]
                        )
                        wandb.log(
                            {
                                f"scatter_{param}_vs_loss": wandb.plot.scatter(
                                    table,
                                    param,
                                    "eval_loss",
                                    title=f"{param.replace('_', ' ').title()} vs Eval Loss",
                                )
                            }
                        )

            # 3. Parameter importance bar chart
            if imp_df is not None and not imp_df.empty:
                imp_data = [
                    [h, i]
                    for h, i in zip(imp_df["hyperparameter"], imp_df["importance"])
                ]
                imp_table = wandb.Table(
                    data=imp_data, columns=["hyperparameter", "importance"]
                )
                wandb.log(
                    {
                        "parameter_importance": wandb.plot.bar(
                            imp_table,
                            "hyperparameter",
                            "importance",
                            title="Hyperparameter Importance (Optuna Fanova)",
                        )
                    }
                )

            # 4. Optimization progress line chart
            progress_data = [[i, loss] for i, loss in enumerate(df["eval_loss"])]
            progress_table = wandb.Table(
                data=progress_data, columns=["trial", "eval_loss"]
            )
            wandb.log(
                {
                    "optimization_progress": wandb.plot.line(
                        progress_table,
                        "trial",
                        "eval_loss",
                        title="Optimization Progress",
                    )
                }
            )

            # 5. Log PNG images if they exist
            png_files = {
                "optuna_optimization_history": "optimization_history.png",
                "optuna_param_importances": "param_importances.png",
                "optuna_parallel_coordinates": "parallel_coordinates.png",
                "optuna_intermediate_values": "intermediate_values.png",
            }

            for log_name, filename in png_files.items():
                filepath = f"{self.output_dir}/{filename}"
                try:
                    wandb.log({log_name: wandb.Image(filepath)})
                except:
                    pass  # File might not exist if kaleido not installed

            # 6. Summary statistics
            assert wandb.run is not None
            wandb.run.summary["best_eval_loss"] = study.best_value
            wandb.run.summary["best_trial"] = study.best_trial.number
            wandb.run.summary["n_completed_trials"] = len(completed_trials)
            wandb.run.summary.update(study.best_params)

            if importances:
                wandb.run.summary["most_important_param"] = max(
                    importances.items(), key=lambda x: x[1]
                )[0]

            wandb.finish()
            logger.info("✅ W&B analysis visualizations uploaded")

        except Exception as e:
            logger.warning(f"⚠️ Could not generate W&B visualizations: {e}")

    def hpo(self):
        """Run hyperparameter optimization."""
        self.preliminary_params_validation()

        logger.info("🔧 Pre-processing: Loading base tokenizer and preparing dataset...")
        base_tokenizer = self._get_base_tokenizer()
        self._prepare_dataset_with_tokenizer(base_tokenizer)
        logger.info("✅ Dataset prepared and cached for all trials")

        # Configure pruner
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=2 if self.max_steps_per_trial <= 0 else 0,
            interval_steps=1,
        )

        # Create study
        study = optuna.create_study(
            direction="minimize",
            pruner=pruner,
            study_name=f"llm-hpo-{self.base_model_name.replace('/', '-')}",
        )

        # Run optimization
        logger.info(f"🚀 Starting HPO with {self.n_trials} trials")
        logger.info(
            f"📝 Configuration: RSLoRA={self.use_rslora}, LoftQ={self.use_loftq}"
        )

        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
            catch=(Exception,),  # Continue even if some trials fail
        )

        # Get best results
        best_params = study.best_trial.params
        best_score = study.best_trial.value

        logger.info(f"🏆 Best trial: {study.best_trial.number}")
        logger.info(f"📊 Best eval loss: {best_score:.4f}")
        logger.info(f"⚙️ Best params: {best_params}")

        # Calculate statistics
        completed_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        pruned_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
        ]
        failed_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.FAIL
        ]

        # Save results
        results = {
            "best_trial_number": study.best_trial.number,
            "best_params": best_params,
            "best_eval_loss": best_score,
            "n_trials": len(study.trials),
            "completed_trials": len(completed_trials),
            "pruned_trials": len(pruned_trials),
            "failed_trials": len(failed_trials),
            "configuration": {
                "base_model": self.base_model_name,
                "use_rslora": self.use_rslora,
                "use_loftq": self.use_loftq,
                "max_seq_length": self.max_seq_length,
            },
        }

        with open(f"{self.output_dir}/best_hparams.json", "w") as f:
            json.dump(results, f, indent=4)

        logger.info(
            f"✅ Best hyperparameters saved to {self.output_dir}/best_hparams.json"
        )
        logger.info(
            f"📈 Trials summary: {len(completed_trials)} completed, {len(pruned_trials)} pruned, {len(failed_trials)} failed"
        )

        # Generate comprehensive analysis and visualizations
        self._generate_analysis_visualizations(study)

        return best_params, best_score

# Example usage:
# if __name__ == "__main__":
#     import os
#     from pathlib import Path
#     from dotenv import load_dotenv
#     from your_module import ModelHandler, DatasetHandler  # Import your existing classes
#
#     load_dotenv()
#
#     # Initialize optimizer - now matches your pipeline structure!
#     optimizer = LLMHyperparameterOptimizer(
#         # Dataset parameters
#         dataset_handler_class=DatasetHandler,  # Pass the class itself
#         dataset_path="path/to/your/dataset",
#         formatted_dataset_path=Path("./tokenized_dataset_hpo"),
#         num_cpus=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
#
#         # Model parameters
#         model_loader_class=ModelHandler,  # Pass the class itself
#         base_model_name="unsloth/llama-3-8b-bnb-4bit",
#         chat_template="llama-3",
#         max_seq_length=2048,
#         use_rslora=True,
#         use_loftq=False,
#
#         # HPO parameters
#         n_trials=20,
#         num_train_epochs=2,
#         max_steps_per_trial=100,  # Limit steps for faster HPO
#         output_dir="./hpo_results",
#
#         # Training parameters (fixed across trials)
#         per_device_train_batch_size=2,
#         gradient_accumulation_steps=4,
#         warmup_steps=10,
#         logging_steps=10,
#         eval_steps=0.1,
#     )
#
#     # Run HPO
#     best_params, best_score = optimizer.hpo()
#
#     print(f"\n🎯 Best hyperparameters found:")
#     for key, value in best_params.items():
#         print(f"  {key}: {value}")
#     print(f"\n📉 Best validation loss: {best_score:.4f}")
