# train-model.py --data_dir <data_dir> [--model_file <model_file>]
#                [--pretrained_model_file <pre_model_file>]
#                [--batch-size 32] [--epochs 300] [--unfreeze 20]
#
# Train a DL model on the data in <data_dir>. The <data_dir> should have a directory
# for each class of interest with images from those classes in the class directories.
# If a pretrained model is available, the file can be provided using the
# --pretrained_model_file option. The trained model is written to <model_file>.
# Default is './model.keras'. The number of non-normalization layers to unfreeze
# for training can be specified by the --unfreeze option. Batch size and epochs can
# also be provided.
#
# Uses EfficientNetB3 designed for 300x300 images.
#
# Author: Lawrence Holder, Washington State University

import os
import sys
import argparse
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet import EfficientNetB3
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

IMAGE_SIZE = 300

def initialize_data_generators(data_dir, batch_size):
    # Training set
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=batch_size,
        #class_mode='binary'  # use 'categorical' if you have more than two classes
        class_mode='categorical',
        subset='training'
    )
    # Validation set
    validation_datagen = ImageDataGenerator(
        validation_split=0.2
    )
    validation_generator = validation_datagen.flow_from_directory(
        data_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=batch_size,
        #class_mode='binary'  # use 'categorical' if you have more than two classes
        class_mode='categorical',
        shuffle=False,
        subset='validation'
    )
    # For evaluating model on all data
    all_datagen = ImageDataGenerator()
    all_generator = all_datagen.flow_from_directory(
        data_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=batch_size,
        #class_mode='binary'  # use 'categorical' if you have more than two classes
        class_mode='categorical',
        shuffle=False
    )
    return train_generator, validation_generator, all_generator

def train_model(train_generator, validation_generator, batch_size, epochs, unfreeze, pretrained_model_path=None):
    if pretrained_model_path:
        print(f"Loading pretrained model from {pretrained_model_path}")
        model = keras.models.load_model(pretrained_model_path)
    else: # Build new model
        # Load the EfficientNet model, pretrained on ImageNet
        base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
        # Freeze the base model
        base_model.trainable = False
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        #x = BatchNormalization()(x)
        #top_dropout_rate = 0.2
        #x = Dropout(top_dropout_rate, name="top_dropout")(x)
        x = Dense(1024, activation='relu')(x)
        #output = Dense(1, activation='sigmoid')(x)  # Use 'softmax' if you have more than two classes
        output = Dense(2, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=output)

    # Unfreeze the top few layers while leaving BatchNorm layers frozen
    for layer in model.layers[-unfreeze:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True
    
    # Compile the model
    #optimizer = keras.optimizers.Adam(learning_rate=1e-3) # set learning_rate (default 1e-3)
    #model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])  # Use 'categorical_crossentropy' for more than two classes
    #model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])  # Use 'categorical_crossentropy' for more than two classes
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # Compute class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights = dict(enumerate(class_weights))
    
    # Train the model
    model.fit(
        train_generator,
        class_weight=class_weights,
        steps_per_epoch=train_generator.n // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.n // batch_size,
        verbose=2
    )
    return model

def evaluate_model(model, validation_generator, all_generator):
    # Evaluate the model on the validation set
    print('\n*** Validation Set Eval ***')
    #loss, accuracy = model.evaluate(validation_generator, verbose=2)
    #print(f'Validation accuracy: {accuracy * 100:.2f}%')
    # Predict the label of the validation set
    val_predictions = model.predict(validation_generator, verbose=2)
    # The model.predict() method returns probabilities for each class.
    # To convert these to actual class predictions, you need to threshold
    # them at 0.5 for binary classification (or take the argmax for multi-class classification).
    #val_predictions = np.round(val_predictions).astype(int)
    val_predictions = np.argmax(val_predictions, axis=1)
    # Retrieve the true labels of the validation set
    val_labels = validation_generator.classes
    # Generate reports
    target_names=validation_generator.class_indices
    print(classification_report(val_labels, val_predictions, target_names=target_names))
    cm = confusion_matrix(val_labels, val_predictions)
    tp, tn, fp, fn = cm[0][0], cm[1][1], cm[1][0], cm[0][1]
    sensitivity = float(tp) / float(tp+fn)
    specificity = float(tn) / float(tn+fp)
    print(target_names)
    print(cm)
    print('sensitivity = {:.2f}'.format(sensitivity))
    print('specificity = {:.2f}'.format(specificity))

    # Evaluate model on all data
    print('\n*** All Data Eval ***')
    all_predictions = model.predict(all_generator, verbose=2)
    all_predictions = np.argmax(all_predictions, axis=1)
    all_labels = all_generator.classes
    # Generate reports
    target_names=all_generator.class_indices
    print(classification_report(all_labels, all_predictions, target_names=target_names))
    cm = confusion_matrix(all_labels, all_predictions)
    tp, tn, fp, fn = cm[0][0], cm[1][1], cm[1][0], cm[0][1]
    sensitivity = float(tp) / float(tp+fn)
    specificity = float(tn) / float(tn+fp)
    print(target_names)
    print(cm)
    print('sensitivity = {:.2f}'.format(sensitivity))
    print('specificity = {:.2f}'.format(specificity))

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', dest='data_dir', type=str, required=True)
    parser.add_argument('--model_file', dest='model_file', type=str, default='./model.keras')
    parser.add_argument('--pretrained_model_file', dest='pretrained_model_file', type=str)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--epochs', dest='epochs', type=int, default=300)
    parser.add_argument('--unfreeze', dest='unfreeze', type=int, default=20)
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    data_dir = args.data_dir
    batch_size = args.batch_size
    epochs = args.epochs
    model_file = args.model_file
    unfreeze = args.unfreeze
    pretrained_model_file = args.pretrained_model_file
    train_generator, validation_generator, all_generator = initialize_data_generators(data_dir, batch_size)
    if pretrained_model_file and os.path.exists(pretrained_model_file):
        model = train_model(train_generator, validation_generator, batch_size, epochs, unfreeze, pretrained_model_file)
    else:
        model = train_model(train_generator, validation_generator, batch_size, epochs, unfreeze)
    model.save(model_file)
    evaluate_model(model, validation_generator, all_generator)
    return

if __name__ == '__main__':
    main()
