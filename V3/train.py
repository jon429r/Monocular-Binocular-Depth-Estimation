from Dataset import Kitti
import torch
import torch.nn as nn

#training model
def train_loop(train_dataloader, model, loss_fn, optimizer):
    for batch_idx, (data, target) in enumerate(train_dataloader):
        try:
            model.train()
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item()}")

        except Exception as e:
            '''print(f"Error in batch {batch_idx}: {e}")
            print("Batch Content:")
            print("Data:", data)
            print("Target:", target)'''
            # Add any additional information that might help diagnose the issue

            # Optionally, break the loop to stop training and investigate the issue
            break



def test_loop(test_dataloader, model, loss):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target_list) in enumerate(test_dataloader):
            # Assuming data is a tensor of images
            outputs = model(data)
            
            # Convert the list of labels to a tensor
            target = torch.tensor(target_list)
            
            # Assuming your model produces class probabilities, use argmax to get predictions
            _, predicted = torch.max(outputs.data, 1)
            
            # Update total and correct counts
            total += target.size(0)
            correct += (predicted == target).sum().item()

    # Calculate accuracy
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')