import torch

from config import EPOCHS, BATCH_SIZE, T, training_data_path
from models.unet import UNet
from diffusion.forward import forward
from data_loaders.load_mnist import get_mnist_loader


model = UNet()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
criterion = torch.nn.MSELoss()

log_every = 1000  # print every 100 steps
step = 0

train_loader = get_mnist_loader(training_data_path, batch_size=BATCH_SIZE)

for epoch in range(EPOCHS):
    for (x0, ) in train_loader:
        B = x0.shape[0]
        t = torch.randint(0, T, (B,))  # random timestep for each image

        x_t, eps = forward(x0, t)
        eps_pred = model(x_t, t)

        loss = criterion(eps_pred, eps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        step += 1
        if step % log_every == 0:
            print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")


torch.save(model.state_dict(), 'model_dict/new.pth')