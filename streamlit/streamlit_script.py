from turtle import width
import streamlit as st
import numpy as np
from PIL import Image
import io
from streamlit_option_menu import option_menu

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow import where

model = load_model('../saved_model/tansfer_model.h5', compile = True)
class_names = ["Pas de voiture", "Voiture"]

st.sidebar.image("../image/logo.png")
# st.sidebar.write('Hello this is a small projet for our deep learning class')
st.sidebar.write("Eliel KATCHE  &  Sven LOTHE")


selected = option_menu(
  menu_title = 'Deep learning Project ',
  options = ["Home","Try it !", "Under the hood",]
)
if selected == "Home":
  st.header('What is this project about')

  st.markdown('We were asked to design a model that is able to detect car in a provided image.   \n  \
      As, we were working with computing power constrains, we decided to use a \
      pre trained  classification Deep learning architecture that offer one of a best accuracy score and \
          that takes minimum time to train, thus, we decided to work with MobileNetV2')

  st.markdown("[Check the references if you are interested](https://keras.io/api/applications/) ")


  st.subheader('Task and Objectives')

  st.markdown('Our goal is to dectect whether an image depicts a car or not.  \n\
      Therefore, we will train our model with a datasat containg images that depicts car and images that does not. \
      Because our GPU ressources are limited, we have found a dataset with small enough images of 64 * 64 pixels.')

  st.markdown("[Link to the datatset ](https://www.kaggle.com/datasets/brsdincer/vehicle-detection-image-set)")

if selected =="Try it !": 
  st.subheader('Try it out !')

  # Code for uploading and checking image
  user_image = None
  user_image = st.file_uploader( "Upload an image from your computer", 
                    type=['png', 'jpg'], 
                    accept_multiple_files=False, 
                    key=None, 
                    help=None, 
                    on_change=None, # add here what to do with the upload
                    args=None, 
                    kwargs=None, 
                    disabled=False)

  if user_image:
    st.write("Here is your image: ")
    st.image(user_image)


    byte_image = Image.open(io.BytesIO(user_image.getvalue()))
    try : 
      resized_image = image.smart_resize(byte_image, (224, 224))
      array_image = image.img_to_array(resized_image)
      model_compatible_image = np.expand_dims(array_image, axis=0)
      st.write("Our model prediction is :")
      answer = model.predict(model_compatible_image)
      predictions = where(answer[0] < 0, 0, 1)
      output_class = class_names[predictions.numpy()[0]]
      st.write(output_class)
    except:
      st.warning("Please choose another image, the current image is not correct")




if selected == "Under the hood" : 

  st.header('What is under the hood ?')

  st.markdown('We began with import, preprocessing and split of the data \
    then displayed them to get a better understanding \
      we imported a pretrained model for transfer learning \
        then fine tuned it to better fit our problem and data \
          finaly we tested it and restart fine tuning until results were good enough')

  st.subheader('Data import and preprocessing')

  st.write('Here are the libraries we used for the preprocessing step')
  imports = '''
  import numpy as np
  from PIL import Image
  import seaborn as sns
  from pathlib import Path
  import tensorflow as tf
  '''
  st.code(imports, language='python')


  st.markdown("The first step of this project was to download our dataset and split it \
      into training testing and validation data. For this ")


  st.markdown('We downloaded the data from [here](https://www.kaggle.com/datasets/brsdincer/vehicle-detection-image-set) \
      as previously mentioned and then we declared our data directory in our notebook and ensure the path to be valid for anyone. \
        To do that, we used Path from pathlib library to get working directory and append it with a relative path.')

  reformated_path = """
  cwd = Path.cwd()
  relative_data_path = "image/data"
  data_dir = cwd / relative_data_path

  >>> WindowsPath('d:/Formation/IASchool/DeepLearning/car_detection/image/data')
  """
  st.code(reformated_path, language='python')


  train_dataset = '''
  train_dataset = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      labels="inferred",
      label_mode='int',
      class_names=None,
      color_mode='rgb',
      batch_size=32,
      image_size=(64, 64),
      shuffle=True,
      seed=42,
      validation_split=0.2,
      subset="training",
      interpolation='bilinear',
      follow_links=False,
      crop_to_aspect_ratio=False,
  )

  >>> Found 9999 files belonging to 2 classes.
  >>> Using 8000 files for training.

  '''
  st.code(train_dataset, language='python')


  test_dataset = '''
  validation_df = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      labels="inferred",
      label_mode='int',
      class_names=None,
      color_mode='rgb',
      batch_size=32,
      image_size=(64, 64),
      shuffle=True,
      seed=42,
      validation_split=0.2,
      subset="validation",
      interpolation='bilinear',
      follow_links=False,
      crop_to_aspect_ratio=False,
  )

  >>> Found 9999 files belonging to 2 classes.
  >>> Using 1999 files for validation.
  '''
  st.code(test_dataset, language='python')


  code = """
  val_batches = tf.data.experimental.cardinality(validation_df)
  test_dataset = validation_df.take(val_batches // 5)
  validation_dataset = validation_df.skip(val_batches // 5)

  print(val_batches)
  >>> tf.Tensor(63, shape=(), dtype=int64) 
  # we spared 63 batches of 32 images for validation and testing

  print(len(test_dataset))
  >>> 12 # 12 batches will be used as test data

  print(len(validation_dataset))
  >>> 51 # 51 batches will be used as validation data

  """
  st.code( code , language='python')


  total_batch = '''
  print('Number of training batches: %d' % tf.data.experimental.cardinality(train_dataset))
  print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
  print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

  >>> Number of training batches: 250
  >>> Number of validation batches: 51
  >>> Number of test batches: 12
  '''
  st.code( total_batch , language='python')


  class_names = '''
  class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
  print(class_names)

  >>> ['non-vehicles' 'vehicles']
  '''
  st.code( class_names , language='python')


  dataset_image = '''
  # display one bach of each dataset (training, test_dataset and validation) 
  for images, labels in train_dataset.take(1):
    plt.figure(figsize=(16, 8))
    plt.suptitle("Training set")
    for i in range(10):
      ax = plt.subplot(2, 5, i + 1)
      plt.imshow(images[i].numpy().astype("uint8"))
      plt.title(class_names[labels[i]])
      plt.axis("off")
    plt.show()

  '''
  st.code( dataset_image , language='python')
  st.image('../notebook_output/show_image_data.png')

  train_set_img= '''

  for images, labels in test_dataset.take(1):
    plt.figure(figsize=(16, 8))
    plt.suptitle("Testing set")
    for i in range(10):
      ax = plt.subplot(2, 5, i + 1)
      plt.imshow(images[i].numpy().astype("uint8"))
      plt.title(class_names[labels[i]])
      plt.axis("off")
    plt.show()

  '''
  st.code( train_set_img , language='python')
  st.image('../notebook_output/test_set.png')


  validation_set_img = """
  for images, labels in validation_dataset.take(1):
    plt.figure(figsize=(16, 8))
    plt.suptitle("Validation set")
    for i in range(10):
      ax = plt.subplot(2, 5, i + 1)
      plt.imshow(images[i].numpy().astype("uint8"))
      plt.title(class_names[labels[i]])
      plt.axis("off")
    plt.show()
  """
  st.code( validation_set_img , language='python')
  st.image('../notebook_output/validation_set.png')

  autotuning = '''
  AUTOTUNE = tf.data.AUTOTUNE

  train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
  validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
  test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
  '''

  st.markdown("Here we use buffered prefetching to load images from disk without having I/O become blocking.\
      you can find some explanations [here](https://www.tensorflow.org/guide/data_performance)" )
  st.code(autotuning, language='python')


  st.subheader('Transfer learning ')

  model_import = """
  pretrained_model = tf.keras.applications.MobileNetV2(weights='imagenet', input_shape=(224, 224, 3), include_top=False)
  """
  model_layers = '''
  for layer in pretrained_model.layers:
          layer.trainable=False
  '''
  st.code( model_import , language='python')
  st.code( model_layers , language='python')

  layer_construnction= '''
  model = pretrained_model.output
  model = tf.keras.layers.GlobalAveragePooling2D()(model)
  model = tf.keras.layers.Dropout(0.2)(model)
  model = Dense(1)(model)
  model = tf.keras.Model(inputs = pretrained_model.input , outputs = model)
  '''
  st.code( layer_construnction , language='python')

  model_compile = '''
  base_learning_rate = 0.0001

  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
  '''

  model_fit= '''
  early_stop = tf.keras.callbacks.EarlyStopping(
      monitor='val_loss',
      min_delta=0,
      patience=3,
      verbose=0,
      mode='auto',
      baseline=None,
      restore_best_weights=False
  )

  initial_epochs = 6

  history = model.fit(train_dataset,
                      epochs=initial_epochs,
                      validation_data=validation_dataset,
                      callbacks=[early_stop])
  '''
  st.code( model_compile , language='python')
  st.code( model_fit , language='python')

  st.write("Here is how our model learned model summary")

  plot_learning = """

  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  plt.figure(figsize=(8, 8))
  plt.subplot(2, 1, 1)
  plt.plot(acc, label='Training Accuracy')
  plt.plot(val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.ylabel('Accuracy')
  plt.ylim([min(plt.ylim()),1])
  plt.title('Training and Validation Accuracy')

  plt.subplot(2, 1, 2)
  plt.plot(loss, label='Training Loss')
  plt.plot(val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.ylabel('Cross Entropy')
  plt.ylim([0,1.5])
  plt.title('Training and Validation Loss')
  plt.xlabel('epoch')
  plt.show()

  """
  st.code( plot_learning , language='python')
  st.image('../notebook_output/final_model.png')

  st.subheader('Evaluation')

  test_batch = """
  image_batch, label_batch = test_dataset.as_numpy_iterator().next()
  predictions = model.predict_on_batch(image_batch).flatten()
  predictions = tf.where(predictions < 0, 0, 1)

  plt.figure(figsize=(16, 16))
  for i in range(32):
    ax = plt.subplot(6, 6, i + 1)
    plt.imshow(image_batch[i].astype("uint8"))
    plt.title(class_names[predictions[i]])
    plt.axis("off")

  """

  st.code( test_batch , language='python')
  st.image('../notebook_output/test_batch.png')


  class_report_batch= '''

  print(classification_report(predictions,label_batch))


                precision    recall  f1-score   support

           0       0.92      1.00      0.96        12
           1       1.00      0.95      0.97        20

    accuracy                           0.97        32
   macro avg       0.96      0.97      0.97        32
weighted avg       0.97      0.97      0.97        32


  '''

  confusion_matrix_batch= """
  cf_matrix = confusion_matrix(predictions, label_batch)
  group_names = ['True Neg','False Pos','False Neg','True Pos']
  group_counts = ["{0:0.0f}".format(value) for value in
                  cf_matrix.flatten()]
  labels = [f"{v1}\n{v2}" for v1, v2 in
            zip(group_names,group_counts)]

  labels = np.asarray(labels).reshape(2,2)

  sns.heatmap(cf_matrix, annot=labels, fmt="")
  """
  st.code( class_report_batch , language='python')
  st.code( confusion_matrix_batch, language='python')
  st.image('../notebook_output/confusion_matrix_batch.png')

  global_eval='''
  all_preds=[]
  all_labels=[]
  for x in range(len(test_dataset)) : 
      for image_batch, label_batch in test_dataset.as_numpy_iterator():
          pred = model.predict_on_batch(image_batch).flatten()
          preds = tf.where(pred < 0, 0, 1)
          all_preds.extend(preds)
          all_labels.extend(label_batch)

  print(classification_report(all_preds,all_labels))


                precision    recall  f1-score   support

            0       0.97      0.96      0.97      7505
            1       0.97      0.97      0.97      7983

      accuracy                           0.97     15488
    macro avg       0.97      0.97      0.97     15488
  weighted avg       0.97      0.97      0.97     15488

  '''
  st.code(global_eval, language='python')

  gloabl_confusion_matrix ='''

  cf_matrix = confusion_matrix(all_preds, all_labels)
  group_names = ['True Neg','False Pos','False Neg','True Pos']
  group_counts = ["{0:0.0f}".format(value) for value in
                  cf_matrix.flatten()]
  labels = [f"{v1}\n{v2}" for v1, v2 in
            zip(group_names,group_counts)]

  labels = np.asarray(labels).reshape(2,2)

  sns.heatmap(cf_matrix, annot=labels, fmt="")
  '''

  st.code(gloabl_confusion_matrix, language='python')
  st.image('../notebook_output/global_confusion.png')