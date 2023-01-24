import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D, Add, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

img_data_df = pd.read_csv('image_data.csv')[:500]

def load_img(path, res):
    img = tf.keras.utils.load_img(path)
    img = tf.keras.utils.img_to_array(img)
    img = tf.image.resize(img, (400,600))
    img = tf.convert_to_tensor(img)
    if res == 'hr':
        img = img / 255
        img = img.numpy()
    elif res == 'lr':
        img = img / 255
        img = img.numpy()
    return img



low_list = []
high_list = []
low_list_image_path = []
high_list_image_path = []
for i in img_data_df['low_res']:
    pp = 'low res/' + i
    low_list_image_path.append(pp)
    temp_tensor_low = load_img(pp, 'lr')
    low_list.append(temp_tensor_low)
for i in img_data_df['high_res']:
    p = 'high res/' + i
    high_list_image_path.append(p)
    temp_tensor = load_img(p, 'hr')
    high_list.append(temp_tensor)

for i in img_data_df['low_res']:
    p = 'low res/' + i[:-6] + '_4.jpg'
    low_list_image_path.append(p)
    temp_tensor_low = load_img(p, 'lr')
    low_list.append(temp_tensor_low)
for i in img_data_df['high_res']:
    p = 'high res/' + i
    high_list_image_path.append(p)
    temp_tensor = load_img(p, 'hr')
    high_list.append(temp_tensor)

for i in img_data_df['low_res']:
    ppp = 'low res/' + i[:-6] + '_6.jpg'
    low_list_image_path.append(ppp)
    temp_tensor_low = load_img(ppp, 'lr')
    low_list.append(temp_tensor_low)
for i in img_data_df['high_res']:
    p = 'high res/' + i
    high_list_image_path.append(p)
    temp_tensor = load_img(p, 'hr')
    high_list.append(temp_tensor)



c = 1
for i in zip(low_list[:3], high_list[:3]):
    fig, axs = plt.subplots(2,1,figsize=(6, 8))
    axs[0].imshow(i[0])
    axs[0].title.set_text('Low')
    axs[1].imshow(i[1])
    axs[1].title.set_text('High')
    plt.tight_layout()
    # plt.savefig('initial_{}_2'.format(c))
    # plt.savefig('initial_{}_4'.format(c))
    # plt.savefig('initial_{}_6'.format(c))
    c+=1
    plt.show()

# p = {'low_res_paths': super_low_list_image_path,
#      'high_res_paths': high_list_image_path}
# p = {'low_res_paths': super_super_low_list_image_path,
#      'high_res_paths': high_list_image_path}
p = {'low_res_paths': low_list_image_path,
     'high_res_paths': high_list_image_path}

p = pd.DataFrame(data = p)

p, test = train_test_split(p, test_size=0.2)

batch_size = 4
original_shape = (400,600)

train_datagen = ImageDataGenerator(rescale = 1/255, validation_split = 0.3)
test_datagen = ImageDataGenerator(rescale = 1/255, validation_split = 0.3)
eval_datagen = ImageDataGenerator(rescale = 1/255)

train_hiresimage_generator = train_datagen.flow_from_dataframe(
        p,
        x_col = 'high_res_paths',
        target_size = original_shape,
        class_mode = None,
        batch_size = batch_size,
        interpolation='nearest',
        seed = 42,
        subset = 'training')

train_lowresimage_generator = train_datagen.flow_from_dataframe(
        p,
        x_col = 'low_res_paths',
        target_size = original_shape,
        class_mode = None,
        batch_size = batch_size,
        interpolation='nearest',
        seed = 42,
        subset = 'training')

val_hiresimage_generator = test_datagen.flow_from_dataframe(
        p,
        x_col = 'high_res_paths',
        target_size = original_shape,
        class_mode = None,
        batch_size = batch_size,
        interpolation='nearest',
        seed = 42,
        subset = 'validation')

val_lowresimage_generator = test_datagen.flow_from_dataframe(
        p,
        x_col = 'low_res_paths',
        target_size = original_shape,
        class_mode = None,
        batch_size = batch_size,
        interpolation='nearest',
        seed = 42,
        subset='validation')

test_hiresimage_generator = eval_datagen.flow_from_dataframe(
        test,
        x_col = 'high_res_paths',
        target_size = original_shape,
        class_mode = None,
        batch_size = batch_size,
        interpolation='nearest',
        seed = 42)

test_lowresimage_generator = eval_datagen.flow_from_dataframe(
        test,
        x_col = 'low_res_paths',
        target_size = original_shape,
        class_mode = None,
        batch_size = batch_size,
        interpolation='nearest',
        seed = 42)

train_generator = zip(train_lowresimage_generator, train_hiresimage_generator)
val_generator = zip(val_lowresimage_generator, val_hiresimage_generator)
test_generator = zip(test_lowresimage_generator, test_hiresimage_generator)

def imageGenerator(generator):
    for (low_res, hi_res) in generator:
            yield (low_res, hi_res)

train_img_gen = imageGenerator(train_generator)
val_image_gen = imageGenerator(val_generator)

def model():
    SRCNN = tf.keras.Sequential()
    SRCNN.add(Conv2D(filters = 64, kernel_size = (9, 9),
                     activation = 'relu', padding = 'same',
                     input_shape = (None, None, 3)))
    SRCNN.add(Conv2D(filters = 32, kernel_size = (1, 1),
                     activation = 'relu', padding = 'same'))
    SRCNN.add(Conv2D(filters = 3, kernel_size = (5, 5),
                     activation = 'relu', padding = 'same'))
    SRCNN.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0005),
                  loss = 'mean_squared_error',
                  metrics = ['mean_squared_error'])
    return SRCNN

train_len = train_hiresimage_generator.samples
steps_per_epoch = train_len // batch_size
val_len = val_hiresimage_generator.samples
validation_steps = val_len // batch_size

path = 'srcnn_model.h5'
isExist = os.path.exists(path)

if not isExist:
    model = model()
    model.summary()
    checkpoint = tf.keras.callbacks.ModelCheckpoint('srcnn_model.h5',
                                                    verbose = 2,
                                                    save_best_only = True)
    early_stopping = tf.keras.callbacks.EarlyStopping(verbose = 2,
                                                      patience = 10)
    plateau = tf.keras.callbacks.ReduceLROnPlateau(verbose = 2,
                                                   factor = 0.1,
                                                   patience = 5)
    model.fit(
    train_img_gen,
    batch_size = batch_size,
    steps_per_epoch = steps_per_epoch,
    validation_data = val_image_gen,
    validation_steps = validation_steps,
    epochs = 50,
    verbose = 2,
    callbacks = [plateau, early_stopping, checkpoint])

    t = iter(test_generator)
    count = 0
    for i in range(len(test)):
        if count == 15:
            break
        x = next(t)
        x_train, x_test = x[0], x[1]
        model.evaluate(x=x[0], y=x[1], verbose=2)
        count += 1
else:
    model = tf.keras.models.load_model('srcnn_model.h5')
    model.summary()
    t = iter(test_generator)
    count = 0
    for i in range(len(test)):
        if count == 15:
            break
        x = next(t)
        x_train, x_test = x[0], x[1]
        model.evaluate(x=x[0], y=x[1], verbose=2)
        count += 1

sample_datagen = ImageDataGenerator(rescale=1/255, validation_split = 0.3)
s = {'low_res_paths': low_list_image_path[:12],
     'high_res_paths': high_list_image_path[:12]}
# s = {'low_res_paths': super_low_list_image_path[:12],
#      'high_res_paths': high_list_image_path[:12]}
# s = {'low_res_paths': super_super_low_list_image_path[:12],
#      'high_res_paths': high_list_image_path[:12]}

s = pd.DataFrame(data = s)
# sample high res
sample_hiresimage_generator = sample_datagen.flow_from_dataframe(
    s,
    x_col = 'high_res_paths',
    target_size = original_shape,
    class_mode = None,
    batch_size = batch_size,
    interpolation = 'nearest',
    seed = 42,
    subset = 'validation')
# sample low res
sample_lowresimage_generator = sample_datagen.flow_from_dataframe(
    s,
    x_col = 'low_res_paths',
    target_size = original_shape,
    class_mode = None,
    batch_size = batch_size,
    interpolation = 'nearest',
    seed = 42,
    subset = 'validation')
sample_generator = zip(sample_lowresimage_generator, sample_hiresimage_generator)

cc = 1
j = iter(sample_generator)
one = next(j)
img1 = one[0]
sr1 = model.predict(img1, batch_size = batch_size)
img1 = cv2.resize(img1[0], (600, 400))
sr1 = cv2.resize(sr1[0], (600, 400))
fig, axs = plt.subplots(2, 1, figsize = (6, 8))
axs[0].imshow(img1)
axs[0].title.set_text('Low')
axs[1].imshow(sr1)
axs[1].title.set_text('SRCNN')
plt.tight_layout()
#plt.savefig('low_high_pred_{}_2'.format(cc))
#plt.savefig('low_high_pred_{}_4'.format(cc))
#plt.savefig('low_high_pred_{}_6'.format(cc))
cc += 1
plt.show()

two = next(j)
img2 = two[0]
sr2 = model.predict(img2, batch_size = batch_size)
img2 = cv2.resize(img2[0], (600, 400))
sr2 = cv2.resize(sr2[0], (600, 400))
fig, axs = plt.subplots(2, 1, figsize = (6, 8))
axs[0].imshow(img2)
axs[0].title.set_text('Low')
axs[1].imshow(sr2)
axs[1].title.set_text('SRCNN')
plt.tight_layout()
#plt.savefig('low_high_pred_{}_2'.format(cc))
#plt.savefig('low_high_pred_{}_4'.format(cc))
#plt.savefig('low_high_pred_{}_6'.format(cc))
cc += 1
plt.show()

three = next(j)
img3 = three[0]
sr3 = model.predict(img3, batch_size = batch_size)
img3 = cv2.resize(img3[0], (600, 400))
sr3 = cv2.resize(sr3[0], (600, 400))
fig, axs = plt.subplots(2, 1, figsize = (6, 8))
axs[0].imshow(img3)
axs[0].title.set_text('Low')
axs[1].imshow(sr3)
axs[1].title.set_text('SRCNN')
plt.tight_layout()
#plt.savefig('low_high_pred_{}_2'.format(cc))
#plt.savefig('low_high_pred_{}_4'.format(cc))
#plt.savefig('low_high_pred_{}_6'.format(cc))
cc += 1
plt.show()
