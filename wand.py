import wandb

run = wandb.init(project ="example-test", name = "my-run")
for i in range(1000):
    my_val= i**2
