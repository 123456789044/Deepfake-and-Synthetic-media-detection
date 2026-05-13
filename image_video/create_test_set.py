import os
import random
import shutil

# source dataset
source = "faces_dataset"

# destination test dataset
destination = "test_dataset"

real_src = os.path.join(source, "real")
fake_src = os.path.join(source, "fake")

real_dest = os.path.join(destination, "real")
fake_dest = os.path.join(destination, "fake")

# create folders
os.makedirs(real_dest, exist_ok=True)
os.makedirs(fake_dest, exist_ok=True)

# number of test images
NUM_TEST = 2000

# filter only image files
real_files = [f for f in os.listdir(real_src) if f.lower().endswith(('.jpg','.jpeg','.png'))]
fake_files = [f for f in os.listdir(fake_src) if f.lower().endswith(('.jpg','.jpeg','.png'))]

# randomly select images
real_images = random.sample(real_files, NUM_TEST)
fake_images = random.sample(fake_files, NUM_TEST)

# copy real images
for img in real_images:
    shutil.copy(os.path.join(real_src, img), os.path.join(real_dest, img))

# copy fake images
for img in fake_images:
    shutil.copy(os.path.join(fake_src, img), os.path.join(fake_dest, img))

print("Test dataset created successfully!")
print("Real images:", len(real_images))
print("Fake images:", len(fake_images))
print("Total test images:", len(real_images) + len(fake_images))