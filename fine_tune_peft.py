import os
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup, DataCollatorForSeq2Seq
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict
from peft.utils.other import fsdp_auto_wrap_policy
from tqdm import tqdm

def main():
    accelerator = Accelerator()
    model_name_or_path = "google/flan-t5-xxl"
    batch_size = 2
    max_length = 512
    lr = 1e-4
    num_epochs = 1
    train_data = "./data/train.csv"
    test_data = "./data/val.csv"
    peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    checkpoint_name = "chaT5_lora.pt"

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
    model = get_peft_model(model, peft_config)
    accelerator.print(model.print_trainable_parameters())

    dataset = load_dataset('csv', data_files={"train": train_data, "validation": test_data}, cache_dir="./cache")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def preprocess_function(examples):
        inputs = [doc for doc in examples["question"]]
        model_inputs = tokenizer(inputs, max_length=max_length, padding=True, truncation=True)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["answer"], max_length=max_length, padding=True, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    with accelerator.main_process_first():
        processed_datasets = dataset.map(preprocess_function, batched=True, num_proc=16,
                                         remove_columns=dataset["train"].column_names,
                                         load_from_cache_file=False,
                                         desc="Running tokenizer on dataset", )

        train_dataset = processed_datasets["train"]
        eval_dataset = processed_datasets["validation"]

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        train_dataloader = DataLoader(train_dataset, shuffle=True,
                                      collate_fn=data_collator,
                                      batch_size=batch_size,
                                      pin_memory=True)

        eval_dataloader = DataLoader(eval_dataset,
                                     collate_fn=data_collator,
                                     batch_size=batch_size,
                                     pin_memory=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                       num_warmup_steps=0,
                                                       num_training_steps=(len(train_dataloader) * num_epochs), )

        if getattr(accelerator.state, "fsdp_plugin", None) is not None:
            accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(model)

        model, train_dataloader, eval_dataloader, optimizer, lr_scheduler = accelerator.prepare(
            model,
            train_dataloader,
            eval_dataloader,
            optimizer,
            lr_scheduler)

        accelerator.print(model)
        accelerator.state.deepspeed_plugin.zero_stage == 3

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0

            for step, batch in enumerate(tqdm(train_dataloader)):
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss.detach().float()

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if step % 1000 == 0:
                    print("loss: ", loss.detach().float())

                accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                accelerator.save(get_peft_model_state_dict(model,
                                                           state_dict=accelerator.get_state_dict(model)),
                                 checkpoint_name)

            model.eval()
            eval_loss = 0
            eval_preds = []

            for step, batch in enumerate(tqdm(eval_dataloader)):
                with torch.no_grad():
                    outputs = model(**batch)
                    loss = outputs.loss
                    eval_loss += loss.detach().float()

                    preds = accelerator.gather_for_metrics(torch.argmax(outputs.logits, -1)).detach().cpu().numpy()
                    eval_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))

            eval_epoch_loss = eval_loss / len(train_dataloader)
            eval_ppl = torch.exp(eval_epoch_loss)

            train_epoch_loss = total_loss / len(eval_dataloader)
            train_ppl = torch.exp(train_epoch_loss)

            accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

            accelerator.wait_for_everyone()
            accelerator.save(get_peft_model_state_dict(model,
                                                       state_dict=accelerator.get_state_dict(model)),
                             checkpoint_name)

            accelerator.wait_for_everyone()

if __name__ == "__main__":
    main()
