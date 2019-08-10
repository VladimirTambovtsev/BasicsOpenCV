import numpy as np
import matplotlib.pyplot as plt
plt.interactive(False)
from PIL import Image
import cv2
import time

my_list = [1,2,3]
print(type(my_list))     # list

my_array = np.array(my_list)
print(type(my_array))      # numpy.ndarray

print(np.arange(0, 10, 2))    # array([0, 2, 4, 6, 8, 10])

print(np.zeros(shape=(2, 5)))   # [[0., 0., 0., 0., 0.],
                                #  [0., 0., 0., 0., 0.]]

print(np.ones(shape=(2, 5)))    # [[1., 1., 1., 1., 1.],
                                #  [1., 1., 1., 1., 1.]]

np.random.seed(101)     # *random number init*
print(np.random.randint(0, 100, 10))    # [95 11 81 70 63 87 75  9 77 40]
print(np.random.randint(0, 100, 10).max())  # 93
print(np.random.randint(0, 100, 10).mean())  # 40.7
print(np.random.randint(0, 100, 10).reshape(2, 5))  # [[76 95 87  0 73]
                                                    #  [ 8 62 36 83 99]]

mat = np.arange(0, 100).reshape(10, 10)     # 10x10 from 0 - 100
print(mat)

row = 0
col = 1
print(mat[row, col])    # 1
print(mat[4, 6])    # 46
print(mat[:, 0])    # first column - [ 0 10 20 30 40 50 60 70 80 90]


pic = Image.open("poppy.jpeg")
pic_array = np.asarray(pic)     # [[[255 255 255] ..... ]]]
print(pic_array)
plt.imshow(pic_array)
plt.show()

red_pic = pic_array.copy()
red_pic = red_pic[:, :, 0]    # R G B
plt.imshow(red_pic)
plt.show()

green_pic = pic_array.copy()
green_pic = green_pic[:, :, 1]    # R G B
plt.imshow(green_pic)
plt.show()

blue_pic = pic_array.copy()
blue_pic = blue_pic[:, :, 2]    # R G B
plt.imshow(blue_pic)
plt.show()


# with gray scale cmap
red_pic = pic_array.copy()
red_pic = red_pic[:, :, 0]    # R G B
plt.imshow(red_pic, cmap="gray")
plt.show()

green_pic = pic_array.copy()
green_pic = green_pic[:, :, 1]    # R G B
plt.imshow(green_pic, cmap="gray")
plt.show()

blue_pic = pic_array.copy()
blue_pic = blue_pic[:, :, 2]    # R G B
plt.imshow(blue_pic, cmap="gray")
plt.show()


# no red color
no_red = pic_array.copy()
no_red[:, :, 0] = 0    # R G B
plt.imshow(no_red)
plt.show()

# no green color
no_green = pic_array.copy()
no_green[:, :, 1] = 0    # R G B
plt.imshow(no_green)
plt.show()

# no blue color
no_blue = pic_array.copy()
no_blue[:, :, 2] = 0    # R G B
plt.imshow(no_blue)
plt.show()

# no blue & green color
no_blue_and_green = pic_array.copy()
no_blue_and_green[:, :, 1] = 0    # R G B
no_blue_and_green[:, :, 2] = 0    # R G B
plt.imshow(no_blue_and_green)
plt.show()


# matplotlib -> Red Green Blue
# openvcv -> Blue Green Red
img = cv2.imread('poppy.jpeg')
print(type(img))
plt.imshow(img)
plt.show()

fix_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # show as Red Green Blue
plt.imshow(fix_img)
plt.show()


img_gray = cv2.imread('poppy.jpeg', cv2.IMREAD_GRAYSCALE)
plt.imshow(img_gray, cmap='gray')
plt.show()

resized_img = cv2.resize(fix_img, (1000, 400))
plt.imshow(resized_img)
plt.show()

w_ration = 0.8
h_ratio = 0.2

rotated_resized_img = cv2.resize(fix_img, (0, 0), fix_img, w_ration, h_ratio)
plt.imshow(rotated_resized_img)
plt.show()


flipped_img = cv2.flip(fix_img, 0)
plt.imshow(flipped_img)
plt.show()


cv2.imwrite('copy_poppy.jpeg', fix_img)     # create copy of the image in cv2


#   drawing on images
blank_img = np.zeros(shape=(512, 512, 3), dtype=np.int16)
plt.imshow(blank_img)
plt.show()

img2 = cv2.imread('poppy.jpeg', 0)
cv2.circle(img2, (100,100),100,255,-1)
cv2.imwrite('drawed_poppy.jpeg', img2)
# ---

# draw by mouse
img = cv2.imread('poppy.jpeg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
flipped_img = cv2.flip(img_rgb, 0)
plt.imshow(flipped_img)
plt.show()
pt1 = (30, 115)
pt2 = (155, 210)
cv2.rectangle(flipped_img, pt1=pt1, pt2=pt2, color=(255, 0, 0), thickness=5)
plt.imshow(flipped_img)
plt.show()
# ----


# blurring
img = cv2.imread('poppy.jpeg').astype(np.float32) / 255   # load img
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def show_img(img):
    fig = plt.figure(figsize=(12, 10))  # show img
    fig.add_subplot(111).imshow(img)
    plt.show()


show_img(img_rgb)   # show original img

gamma = 0.5   # show with gamma value - lighter
img_with_gamma_v = np.power(img_rgb, gamma)
show_img(img_with_gamma_v)

gamma = 2   # show with gamma value - darker
img_with_gamma_v = np.power(img_rgb, gamma)
show_img(img_with_gamma_v)

kernel = np.ones(shape=(5, 5), dtype=np.float32) / 25   # blurring
dst = cv2.filter2D(img_rgb, -1, kernel)
show_img(dst)

blurred = cv2.blur(img_rgb, ksize=(5, 5))
show_img(blurred)

blurred_img = cv2.GaussianBlur(img_rgb, (5, 5), 10)
show_img(blurred_img)
# ---


# # Video capturing
# cap = cv2.VideoCapture(0)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
# writer = cv2.VideoWriter('supervideo.mp4', cv2.VideoWriter_fourcc(*'XVID'), 20, (width, height))     # XVID - Unix; DIVX - Windows
#
# while True:
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('frame', gray)
#
#     writer.write(frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# writer.release()
# cv2.destroyAllWindows()
# ---


# # Using video files
# cap = cv2.VideoCapture('supervideo.mp4')
#
# if not cap.isOpened():
#     print('File not found')
# while cap.isOpened():
#     ret, frame = cap.read()
#
#     if ret:
#         time.sleep(1/20)    # 20fps
#         cv2.imshow('frame', frame)
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break
#     else:
#         break
# ---


# template matching
img = cv2.imread('poppy.jpeg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_face = cv2.imread('poppy_face.jpeg')
img_face_rgb = cv2.cvtColor(img_face, cv2.COLOR_BGR2RGB)
# eval('sum')
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
for m in methods:
    img_copy = img_rgb.copy()
    method = eval(m)
    res = cv2.matchTemplate(img_copy, img_face_rgb, method)
    min_val, max_val, min_loc, max_loc, = cv2.minMaxLoc(res)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    height, width, channels = img_face_rgb.shape
    bottom_right = (top_left[0] + width, top_left[1] + height)
    cv2.rectangle(img_copy, top_left, bottom_right, (255, 0, 0), 10)
    plt.subplot(121)
    plt.imshow(res)
    plt.title('Heat map of template matching')
    plt.subplot(122)
    plt.imshow(img_copy)
    plt.title('Detection of template')
    plt.suptitle(m)
    plt.show()
# ---


# # Watershed algorithm
# road_img = cv2.imread('road_image.jpg')
# road_img_copy = np.copy(road_img)
# marker_img = np.zeros(road_img.shape[:2], dtype=np.int32)
# segments = np.zeros(road_img.shape, dtype=np.uint8)
# from matplotlib import cm
# cm.tab10(0)
#
#
# def create_rgb(img):
#     return tuple(np.array(cm.tab10(img)[:3])*255)
#
# colors = []
# for i in range(10):
#     colors.append(create_rgb(i))
#
# n_markers = 10 # 0-9
# current_marker = 1
# marks_updated = False
#
#
# def mouse_callback(event, x, y, flags, params):
#     global marks_updated
#     if event == cv2.EVENT_LBUTTONDOWN:
#         cv2.circle(marker_img, (x, y), 10, (current_marker), -1)
#         cv2.circle(road_img_copy, (x, y), 10, colors[current_marker], -1)
#         marks_updated = True
#
# cv2.namedWindow('Road Image')
# cv2.setMouseCallback('Road Image', mouse_callback)
#
# while True:
#     cv2.imshow('Watershed Segments', segments)
#     cv2.imshow('Road Image', road_img_copy)
#     k = cv2.waitKey(1)
#     if k == 27:
#         break
#     elif k == ord('c'):
#         road_img_copy = road_img.copy()
#         marker_img = np.zeros(road_img.shape[:2], dtype=np.int32)
#         segments = np.zeros(road_img.shape, dtype=np.uint8)
#     elif k > 0 and chr(k).isdigit():
#         current_marker = int(chr(k))
#     if marks_updated:
#         marker_img_copy = marker_img.copy()
#         cv2.watershed(road_img, marker_img_copy)
#         segments = np.zeros(road_img.shape, dtype=np.uint8)
#         for color_ind in range(n_markers):
#             segments[marker_img_copy == (color_ind)] = colors[color_ind]
# cv2.destroyAllWindows()
# # ---


# # face detection - Viola-Jones algorithm
# poppy = cv2.imread('poppy.jpeg', 0)
# nadia = cv2.imread('faces/Nadia_Murad.jpg', 0)
# denis = cv2.imread('faces/Denis_Mukwege.jpg', 0)
# solvay = cv2.imread('faces/solvay_conference.jpg', 0)
# plt.imshow(nadia, cmap='gray')
# face_cascade = cv2.CascadeClassifier('faces/haarcascades/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('faces/haarcascades/haarcascade_eye.xml')
#
# def detect_face(img):
#     face_img = img.copy()
#     face_rects = face_cascade.detectMultiScale(face_img)
#     for (x, y, w, h) in face_rects:
#         cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 10)
#     return face_img
#
#
# def adj_detect_face(img):
#     face_img = img.copy()
#     face_rects = face_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)
#     for (x, y, w, h) in face_rects:
#         cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 10)
#     return face_img
#
#
# def detect_eyes(img):
#     face_img = img.copy()
#     eyes_rects = eye_cascade.detectMultiScale(face_img, scaleFactor=1.2, minNeighbors=5)
#     for (x, y, w, h) in eyes_rects:
#         cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 10)
#     return face_img
#
# result = detect_face(denis)
# plt.imshow(result, cmap='gray')
# plt.show()
#
# result = detect_face(nadia)
# plt.imshow(result, cmap='gray')
# plt.show()
#
# result = detect_face(solvay)
# plt.imshow(result, cmap='gray')
# plt.show()
#
# result = adj_detect_face(solvay)
# plt.imshow(result, cmap='gray')
# plt.show()
#
# result = detect_eyes(nadia)
# plt.imshow(result, cmap='gray')
# plt.show()
#
#
# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read(0)
#     frame = detect_face(frame)
#     cv2.imshow('Video Face Detect', frame)
#     k = cv2.waitKey(1)
#     if k == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()
# ---


# cars numbers detector
car1 = cv2.imread('cars/car_plate.jpg')
plate_cascade = cv2.CascadeClassifier('cars/haarcascades/haarcascade_russian_plate_number.xml')


def display(img):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(new_img)
    plt.show()


def detect_plate(img):
    plate_img = img.copy()
    plate_rects = plate_cascade.detectMultiScale(plate_img, scaleFactor=1.3, minNeighbors=3)
    for (x, y, w, h) in plate_rects:
        cv2.rectangle(plate_img, (x, y), (x + w, y + h), (0, 0, 255), 4)
    return plate_img


def detect_and_blur_plate(img):
    plate_img = img.copy()
    roi = img.copy()
    plate_rects = plate_cascade.detectMultiScale(plate_img, scaleFactor=1.3, minNeighbors=3)
    for (x, y, w, h) in plate_rects:
        roi = roi[y: y + h, x: x + w]
        blurred_roi = cv2.medianBlur(roi, 7)
        plate_img[y: y + h, x: x + w] = blurred_roi
        cv2.rectangle(plate_img, (x, y), (x + w, y + h), (0, 0, 255), 4)
    return plate_img


result = detect_plate(car1)
display(result)

result = detect_and_blur_plate(car1)
display(result)
# ---


# # optical flow
# # Parameters for ShiTomasi corner detection (good features to track paper)
# corner_track_params = dict(maxCorners=10, qualityLevel=0.3, minDistance=7, blockSize=7 )
# # Parameters for lucas kanade optical flow
# lk_params = dict(winSize=(200,200), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#
# cap = cv2.VideoCapture(0)
# ret, prev_frame = cap.read()
# prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
#
# # Grabbing the corners & create mask
# prevPts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **corner_track_params)
# mask = np.zeros_like(prev_frame)
#
# while True:
#
#     ret, frame = cap.read()
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Calculate the Optical Flow on the Gray Scale Frame
#     nextPts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prevPts, None, **lk_params)
#
#     # Using the returned status array (the status output)
#     # status output status vector (of unsigned chars); each element of the vector is set to 1 if
#     # the flow for the corresponding features has been found, otherwise, it is set to 0.
#     good_new = nextPts[status == 1]
#     good_prev = prevPts[status == 1]
#
#     # Use ravel to get points to draw lines and circles
#     for i, (new, prev) in enumerate(zip(good_new, good_prev)):
#         x_new, y_new = new.ravel()
#         x_prev, y_prev = prev.ravel()
#
#         # Lines will be drawn using the mask created from the first frame
#         mask = cv2.line(mask, (x_new, y_new), (x_prev, y_prev), (0, 255, 0), 3)
#
#         # Draw red circles at corner points
#         frame = cv2.circle(frame, (x_new, y_new), 8, (0, 0, 255), -1)
#
#     # Display the image along with the mask we drew the line on.
#     img = cv2.add(frame, mask)
#     cv2.imshow('frame', img)
#
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break
#
#     # Now update the previous frame and previous points
#     prev_gray = frame_gray.copy()
#     prevPts = good_new.reshape(-1, 1, 2)
#
# cv2.destroyAllWindows()
# cap.release()
# # ---


# keras basic
from numpy import genfromtxt
data = genfromtxt('nn-data/bank_note_data.txt', delimiter=',')
print('data: \n')
print(data)
labels = data[:, 4]
features = data[:, 0:4]
X = features
y = labels

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.preprocessing import MinMaxScaler      # scale values for perfomance increase
scaler_object = MinMaxScaler()
scaler_object.fit(X_train)

scaled_X_train = scaler_object.transform(X_train)
scaled_X_test = scaler_object.transform(X_test)
scaled_X_train

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(4, input_dim=4, activation='relu'))     # 4 neurons, 1st layer
model.add(Dense(8, activation='relu'))  # 8 neurons, 2nd layer
model.add(Dense(1, activation='sigmoid'))   # 1 output neuron, 3rd layer, binary classification
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])   # config learning
model.fit(scaled_X_train, y_train, epochs=50, verbose=2)   # start learning with 50 steps back propagation


from sklearn.metrics import confusion_matrix, classification_report     # evaluate the model: how good is it
predictions = model.predict_classes(scaled_X_test)
confusion_matrix(y_test, predictions)
print(classification_report(y_test, predictions))

model.save('my_supermodel.h5')  # save model, if it was good

from keras.models import load_model
newmodel = load_model('my_supermodel.h5')    # import built model
newmodel.predict_classes(scaled_X_test)     # try imported model
# ---


# # mnist cnn with keras
# from keras.datasets import mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# x_train.shape   #   60 000 images, 28x28
# single_image = x_train[0]
# plt.imshow(single_image, cmap='gray')
# plt.show()
#
# from keras.utils.np_utils import to_categorical
# y_cat_test = to_categorical(y_test, 10)
# y_cat_train = to_categorical(y_train, 10)
#
# x_train = x_train / x_train.max()   # x / 255 - normalize color from 0-255 to 0-1
# x_test = x_test / x_test.max()
#
# x_train = x_train.reshape(60000, 28, 28, 1)     # add color channel 0-1
# x_test = x_test.reshape(10000, 28, 28, 1)
#
#
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, BatchNormalization
# # set model
# model = Sequential()
# model.add(Conv2D(filters=36,
#                  kernel_size=(3, 3),
#                  padding='same',
#                  input_shape=(28, 28, 1),
#                  activation='relu'))    # 1st layer, 36 neurons, convolutional
# model.add(Conv2D(filters=36,
#                  kernel_size=(3, 3),
#                  padding='same',
#                  activation='relu'))    # 2nd layer, 36 neurons, convolutional
# model.add(MaxPooling2D(pool_size=(2, 2)))   # 3rd layer, pooling
# model.add(Conv2D(filters=36,
#                  kernel_size=(3, 3),
#                  padding='same',
#                  activation='relu'))    # 4th layer, 36 neurons, convolutional
# model.add(Conv2D(filters=36,
#                  kernel_size=(3, 3),
#                  padding='same',
#                  activation='relu'))    # 5th layer, 36 neurons, convolutional
# model.add(MaxPooling2D(pool_size=(2, 2)))   # 6rd layer, pooling
# model.add(Dropout(rate=0.25))    # 7th layer
# model.add(Flatten())    # 8th layer, 2d image to 1d
# model.add(Dense(units=512))     # 9th layer
# model.add(BatchNormalization())     # 10th layer, normalization
# model.add(Activation('relu'))   # 11th layer, ?
# model.add(Dropout(rate=0.25))    # 12th layer
# model.add(Dense(10, activation='softmax'))  # 13th layer, 10 neurons, final layer
#
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # config learning
#
#
# from keras.preprocessing.image import ImageDataGenerator
# datagen = ImageDataGenerator(
#     featurewise_center=False,  # set input mean to 0 over the dataset
#     samplewise_center=False,  # set each sample mean to 0
#     featurewise_std_normalization=False,  # divide inputs by std of the dataset
#     samplewise_std_normalization=False,  # divide each input by its std
#     zca_whitening=False,  # apply ZCA whitening
#     rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
#     zoom_range=0.1,  # Randomly zoom image
#     width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#     height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
#     horizontal_flip=False,  # randomly flip images
#     vertical_flip=False)  # randomly flip images
# datagen.fit(x_train)
#
#
# # Decay learning rate
# from keras.callbacks import ReduceLROnPlateau
# learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
#                                             patience=3,
#                                             verbose=1,
#                                             factor=0.5,
#                                             min_lr=0.00001)
#
# batch_size = 32
# epochs = 2  # optimal is 10
# train_history = model.fit_generator(datagen.flow(x_train, y_cat_train, batch_size=batch_size),
#                                     epochs=epochs,
#                                     verbose=2, steps_per_epoch=x_train.shape[0] # batch_size
#                                     , callbacks=[learning_rate_reduction])
#
#
#
# #  Show CNN train history: 30 epochs
# plt.plot(train_history.history['acc'])
# plt.plot(train_history.history['val_acc'])
# epoch_num = len(train_history.epoch)
# final_epoch_train_acc = train_history.history['acc'][epoch_num - 1]
# final_epoch_validation_acc = train_history.history['val_acc'][epoch_num - 1]
# plt.text(epoch_num, final_epoch_train_acc, 'train = {:.3f}'.format(final_epoch_train_acc))
# plt.text(epoch_num, final_epoch_validation_acc-0.01, 'valid = {:.3f}'.format(final_epoch_validation_acc))
# plt.title('Train History')
# plt.ylabel('accuracy')
# plt.xlabel('Epoch')
# plt.xlim(xmax=epoch_num+1)
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
#
#
# # Show top 6 prediction errors
# def display_errors(errors_index, img_errors, pred_errors, obs_errors):
#     """ This function shows 6 images with their predicted and real labels"""
#     n = 0
#     nrows = 2
#     ncols = 3
#     fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True)
#     for row in range(nrows):
#         for col in range(ncols):
#             error = errors_index[n]
#             ax[row, col].imshow((img_errors[error]).reshape((28, 28)))
#             ax[row, col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error], obs_errors[error]))
#             n += 1
#     plt.show()
#
#
#
# y_pred = model.predict(x_test)
# y_pred_classes = np.argmax(y_pred, axis=1)
# y_true = np.argmax(y_cat_test, axis=1)
#
# errors = (y_pred_classes - y_true != 0)
# y_pred_classes_errors = y_pred_classes[errors]
# y_pred_prob_errors = y_pred[errors]
# y_true_classes_errors = y_true[errors]
# x_validation_errors = x_test[errors]
#
# y_pred_maxProb_errors = np.max(y_pred_prob_errors, axis=1)
# y_true_prob_errors = np.diagonal(np.take(y_pred_prob_errors, y_true_classes_errors, axis=1))
# deltaProb_pred_true_errors = y_pred_maxProb_errors - y_true_prob_errors
# sorted_delaProb_errors = np.argsort(deltaProb_pred_true_errors)
#
# # Top 6 errors
# top6_errors = sorted_delaProb_errors[-6:]
#
# # Show the top 6 errors
# display_errors(top6_errors, x_validation_errors, y_pred_classes_errors, y_true_classes_errors)
#
#
#
# model.evaluate(x_test, y_cat_test)
#
# from sklearn.metrics import classification_report   # evaluate the model
# predictions = model.predict_classes(x_test)
# print(classification_report(y_test, predictions))
