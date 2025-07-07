from dataset_handler import DatasetHandler
from model_handler import ModelHandler

if __name__ == "__main__":
    dh: DatasetHandler = DatasetHandler()
    hf_train_data, hf_eval_data, df_test_data = dh.HF_DATASET_run_pipeline()
    mh: ModelHandler = ModelHandler(
        hf_train_data=hf_train_data,
        hf_eval_data=hf_eval_data,
        df_test_data=df_test_data,
    )
    mh.WB_init()
    mh.MODEL_load()
    mh.TOKENIZER_load()
    mh.HF_SFTTrainer()
    mh.TRAINER_run_training()
    # mh.MODEL_load_ft_model()
    # mh.PEFT_MODEL_infer()
