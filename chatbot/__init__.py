from simpletransformers.conv_ai import ConvAIModel


train_args = {
    "num_train_epochs": 50,
    "save_model_every_epoch": False,
}

!wget https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/gpt_personachat_cache.tar.gz

!mkdir gpt_personachat_cache
!tar -xvzf /content/gpt_personachat_cache.tar.gz -C gpt_personachat_cache

!mkdir cache_dir
# Create a ConvAIModel
model = ConvAIModel("gpt", "gpt_personachat_cache", use_cuda=True, args=train_args)

# Train the model
# model.train_model("/content/train_data.json")

# # Evaluate the model
# model.eval_model()

# Interact with the trained model.
model.interact()