from LLaVA_wrapper.llava_local.train.train import train

if __name__ == "__main__":
    # train(
    #     attn_implementation="eager"
    # )
     
    train(
        attn_implementation="flash_attention_2"
    )
