import tensorflow.keras as keras

def create_model_from_mst(mst):
    if mst['model'] == 'vgg16':
        model = keras.applications.vgg16.VGG16(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=INPUT_SHAPE,
            pooling=None,
            classes=NUM_CLASSES)
    elif mst['model'] == 'resnet50':
        model = keras.applications.ResNet50(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=INPUT_SHAPE,
            pooling=None,
            classes=NUM_CLASSES)
    elif mst['model'] == 'resnet152':
        model = keras.applications.ResNet152(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=INPUT_SHAPE,
            pooling=None,
            classes=NUM_CLASSES)
    elif mst['model'] == 'densenet121':
        model = keras.applications.DenseNet121(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=INPUT_SHAPE,
            pooling=None,
            classes=NUM_CLASSES)
    elif mst['model'] == 'nasnetmobile':
        model = keras.applications.NASNetMobile(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=INPUT_SHAPE,
            pooling=None,
            classes=NUM_CLASSES)
    elif mst['model'] == 'mobilenetv2':
        model = keras.applications.MobileNetV2(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=INPUT_SHAPE,
            pooling=None,
            classes=NUM_CLASSES)
    elif mst['model'] == 'mobilenetv1':
        model = keras.applications.MobileNet(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=INPUT_SHAPE,
            pooling=None,
            classes=NUM_CLASSES)
    elif mst['model'] == 'sanity':
        model = Sequential()
        model.add(Dense(10, activation='relu', input_shape=(4,)))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(3, activation='softmax'))
    model = patch_model(model, mst['lambda_value'])
    return model