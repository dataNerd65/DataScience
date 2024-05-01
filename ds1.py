from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# Create a Sequential model
model = Sequential()

# Add convolutional layer
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# Add pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add second convolutional layer
model.add(Conv2D(32, (3, 3), activation='relu'))

# Add second pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the tensor output from the convolutional layers
model.add(Flatten())

# Add fully connected layer
model.add(Dense(units=128, activation='relu'))

# Add output layer
model.add(Dense(units=10, activation='softmax'))  # 10 units for 10 classes

# Compile the CNN
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Part 2 - Fitting the CNN to the images
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('animal_data/training_set', target_size=(64, 64), batch_size=32, class_mode='categorical')
test_set = test_datagen.flow_from_directory('animal_data/test_set', target_size=(64, 64), batch_size=32, class_mode='categorical')

model.fit_generator(training_set, steps_per_epoch=8000, epochs=25, validation_data=test_set, validation_steps=2000)