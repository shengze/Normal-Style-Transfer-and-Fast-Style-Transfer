from IPython.display import Image, display
#Image('F:/olddesk/python/styling_tech/images/image_view.png')
import cv2
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import PIL.Image
import argparse
import time

#from keras.models import Graph
import vgg16
import vgg16_fastNeural

#vgg16.maybe_download()

def load_image(filename, max_size=None):
    image = PIL.Image.open(filename)
    
    if max_size is not None:
        factor = max_size / np.max(image.size)
        
        size = np.array(image.size) * factor
        size = size.astype(int)
        
        image = image.resize(size, PIL.Image.LANCZOS)
        
    return np.float32(image)
    
    
def save_image(image, filename):
    image = np.clip(image, 0.0, 255.0)
    
    image = image.astype(np.uint8)
    
    with open(filename, 'wb') as file:
        PIL.Image.fromarray(image).save(file, 'jpg')
        
        
def plot_image_big(image, result):
    image = np.clip(image, 0.0, 255.0)
    
    image = image.astype(np.uint8)
    
    display(PIL.Image.fromarray(image))
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(result, image)
    
    
    
def plot_images(content_image, style_image, mixed_image):
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))
    
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    
    smooth = True
    if smooth:
        interpolation = 'sinc'
    else:
        interpolation = 'nearest'
        
    ax = axes.flat[0]
    ax.imshow(content_image / 255.0, interpolation=interpolation)
    ax.set_xlabel("Content")
    
    ax = axes.flat[1]
    ax.imshow(mixed_image / 255.0, interpolation=interpolation)
    ax.set_xlabel("Mixed")
    
    ax = axes.flat[2]
    ax.imshow(style_image / 255.0, interpolation=interpolation)
    ax.set_xlabel("Style")
    
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.show()
    
    
def mean_squared_error(a, b):
    return tf.reduce_mean(tf.square(a - b))
    

def create_content_loss(session, model, content_image, layer_ids):
    feed_dict = model.create_feed_dict(image=content_image)
    
    layers = model.get_layer_tensors(layer_ids)
    
    values = session.run(layers, feed_dict=feed_dict)
    
    with model.graph.as_default():
        layer_losses = []

        for value, layer in zip(values, layers):
            value_const = tf.constant(value)
            
            loss = mean_squared_error(layer, value_const)
            
            layer_losses.append(loss)
            
        total_loss = tf.reduce_mean(layer_losses)
        
    return total_loss
    
    
def gram_matrix(tensor):
    shape = tensor.get_shape()
    
    num_channels = int(shape[3])
    
    matrix = tf.reshape(tensor, shape=[-1, num_channels])
    
    gram = tf.matmul(tf.transpose(matrix), matrix)
    
    return gram
    
    
def create_style_loss(session, model, style_image, layer_ids):
    feed_dict = model.create_feed_dict(image=style_image)
    
    layers = model.get_layer_tensors(layer_ids)
    
    with model.graph.as_default():
        gram_layers = [gram_matrix(layer) for layer in layers]
                       
        values = session.run(gram_layers, feed_dict=feed_dict)
        
        layer_losses = []

        for value, gram_layer in zip(values, gram_layers):
            value_const = tf.constant(value)
            
            loss = mean_squared_error(gram_layer, value_const)
            
            layer_losses.append(loss)
            
        total_loss = tf.reduce_mean(layer_losses)
        
    return total_loss
    
    
def create_denoise_loss(model):
    loss = tf.reduce_sum(tf.abs(model.input[:,1:,:,:] - model.input[:,:-1,:,:])) + \
           tf.reduce_sum(tf.abs(model.input[:,:,1:,:] - model.input[:,:,:-1,:]))
    
    return loss
    
    
def style_transfer(args, fst_training,
                   content_image, style_image, 
                   content_layer_ids, style_layer_ids, 
                   content_img_path, style_img_path, result,
                   model_name,
                   weight_content=1.5, weight_style=10.0, 
                   weight_denoise=0.3, 
                   num_iterations=120, step_size=10.0):
    
    if fst_training:
        train(style_img_path, model_name, args)
    
    else:
        model = vgg16.VGG16()
        
        session = tf.InteractiveSession(graph=model.graph)
        
        print("Content Layers:")
        print(model.get_layer_names(content_layer_ids))
        print()
        
        print("Style Layers:")
        print(model.get_layer_names(style_layer_ids))
        print()
        
        loss_content = create_content_loss(session=session, 
                                           model=model, 
                                           content_image=content_image, 
                                           layer_ids=content_layer_ids)
        
        loss_style = create_style_loss(session=session,
                                       model=model,
                                       style_image=style_image, 
                                       layer_ids=style_layer_ids)
        
        loss_denoise = create_denoise_loss(model)
        
        adj_content = tf.Variable(1e-10, name='adj_content')
        adj_style = tf.Variable(1e-10, name='adj_style')
        adj_denoise = tf.Variable(1e-10, name='adj_denoise')
        
        session.run([adj_content.initializer,
                     adj_style.initializer,
                     adj_denoise.initializer])
        
        update_adj_content = adj_content.assign(1.0 / (loss_content + 1e-10))
        update_adj_style = adj_style.assign(1.0 / (loss_style + 1e-10))
        update_adj_denoise = adj_denoise.assign(1.0 / (loss_denoise + 1e-10))
        
        loss_combined = weight_content * adj_content * loss_content + \
                        weight_style * adj_style * loss_style + \
                        weight_denoise * adj_denoise * loss_denoise
                        
        print(model.input)
        gradient = tf.gradients(loss_combined, model.input)
#        global_step = tf.Variable(0, name='global_step', trainable=False)
#        gradient = tf.train.AdamOptimizer(1e-3).minimize(loss_combined)
#        gradient = tf.train.AdamOptimizer.compute_gradients(loss_combined, model.input)
        run_list = [gradient, update_adj_content, update_adj_style, update_adj_denoise]
    
        #modified init image
        content_resizename = content_img_path
        style_resizename = style_img_path
        standard_size_image = PIL.Image.open(content_resizename)
        resize_image = PIL.Image.open(style_resizename)
        size = np.array(standard_size_image.size)
        size = size.astype(int)
        style_resize = resize_image.resize(size, PIL.Image.LANCZOS)        
        style_resize = np.float32(style_resize)
        mixed_image = np.random.rand(*content_image.shape) + 128
        #mixed_image = (content_image + style_resize)/2
        #mixed_image = content_image
        
        #two plot list
        plot_list1 = []
        
        t_start = time.time()
        t_e1 = 0
        t_e2 = 0
        for i in range(num_iterations):
            feed_dict = model.create_feed_dict(image=mixed_image)
            
            grad, adj_content_val, adj_style_val, adj_denoise_val = session.run(run_list, feed_dict=feed_dict)
            
            grad = np.squeeze(grad)
            
            step_size_scaled = step_size / (np.std(grad) + 1e-8)
            
            mixed_image -= grad * step_size_scaled
            
            mixed_image = np.clip(mixed_image, 0.0, 255.0)
            
            print(". ", end="")
            
            if (i % 10 == 0) or (i == num_iterations - 1):
                print()
                print("Iteration:", i)
                
                msg = "Weight Adj. for Content: {0:.2e}, Style: {1:.2e}, Denoise: {2:.2e}"
                print(msg.format(adj_content_val, adj_style_val, adj_denoise_val))
                
                loss_c = weight_content * adj_content_val + \
                        weight_style * adj_style_val + \
                        weight_denoise * adj_denoise_val
                print('weighted loss:', loss_c)
                if i != 0:
                    plot_list1.append(loss_c)
                
                #print(grad)
                
                plot_images(content_image=content_image,
                            style_image=style_image,
                            mixed_image=mixed_image)
            if i == 100:
                t_e1 = time.time() - t_start
            if i == 300:
                t_e2 = time.time() - t_start

        t_end = time.time() - t_start
        print('execution time 100 iter:', t_e1)
        print('execution time 300 iter:', t_e2)
        print('execution time 500 iter:', t_end)
        print()
        print("Final Image:")
        plot_image_big(mixed_image, result)
        
        plt.plot(plot_list1)
        print('loss list:', plot_list1)
        plt.show()
        
        session.close()
        
        return mixed_image
    
    
def train(style_img_path, model_name, args):
    """main
    :param args:
        argparse.Namespace object from argparse.parse_args().
    """
    # Unpack command-line arguments.
    train_dir = args.train_dir
    style_img_path = args.style_img_path
    model_name = args.model_name
    preprocess_size = args.preprocess_size
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    run_name = args.run_name
    learn_rate = args.learn_rate
    loss_content_layers = args.loss_content_layers
    loss_style_layers = args.loss_style_layers
    content_weights = args.content_weights
    style_weights = args.style_weights
    num_steps_ckpt = args.num_steps_ckpt
    num_pipe_buffer = args.num_pipe_buffer
    num_steps_break = args.num_steps_break
    beta_val = args.beta
    style_target_resize = args.style_target_resize
    upsample_method = args.upsample_method

    # Load in style image that will define the model.
    style_img = imread(style_img_path)
    style_img = imresize(style_img, style_target_resize)
    style_img = style_img[np.newaxis, :].astype(np.float32)

    # Alter the names to include a namescope that we'll use + output suffix.
    loss_style_layers = ['vgg/' + i + ':0' for i in loss_style_layers]
    loss_content_layers = ['vgg/' + i + ':0' for i in loss_content_layers]

    # Get target Gram matrices from the style image.
    with tf.variable_scope('vgg'):
        X_vgg = tf.placeholder(tf.float32, shape=style_img.shape, name='input')
        vggnet = vgg16_fastNeural.vgg16(X_vgg)
    with tf.Session() as sess:
        vggnet.load_weights('libs/vgg16_weights.npz', sess)
        print ('Precomputing target style layers.')
        target_grams = sess.run(get_grams(loss_style_layers),
                                feed_dict={X_vgg: style_img})

    # Clean up so we can re-create vgg connected to our image network.
    print ('Resetting default graph.')
    tf.reset_default_graph()

    # Load in image transformation network into default graph.
    shape = [batch_size] + preprocess_size + [3]
    with tf.variable_scope('img_t_net'):
        X = tf.placeholder(tf.float32, shape=shape, name='input')
        Y = create_net(X, upsample_method)

    # Connect vgg directly to the image transformation network.
    with tf.variable_scope('vgg'):
        vggnet = vgg16_fastNeural.vgg16(Y)

    # Get the gram matrices' tensors for the style loss features.
    input_img_grams = get_grams(loss_style_layers)

    # Get the tensors for content loss features.
    content_layers = get_layers(loss_content_layers)

    # Create loss function
    content_targets = tuple(tf.placeholder(tf.float32,
                            shape=layer.get_shape(),
                            name='content_input_{}'.format(i))
                            for i, layer in enumerate(content_layers))
    cont_loss = fst_content_loss(content_layers, content_targets,
                                    content_weights)
    style_loss = fst_style_loss(input_img_grams, target_grams,
                                   style_weights)
    tv_loss = fst_tv_loss(Y)
    beta = tf.placeholder(tf.float32, shape=[], name='tv_scale')
    loss = cont_loss + style_loss + beta * tv_loss
    with tf.name_scope('summaries'):
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('style_loss', style_loss)
        tf.summary.scalar('content_loss', cont_loss)
        tf.summary.scalar('tv_loss', beta*tv_loss)

    # Setup input pipeline (delegate it to CPU to let GPU handle neural net)
    files = tf.train.match_filenames_once(train_dir + '/train-*')
    with tf.variable_scope('input_pipe'), tf.device('/cpu:0'):
        batch_op = batcher(files, batch_size, preprocess_size,
                                    n_epochs, num_pipe_buffer)

    # We do not want to train VGG, so we must grab the subset.
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                   scope='img_t_net')

    # Setup step + optimizer
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learn_rate).minimize(loss, global_step,
                                                            train_vars)

    # Setup subdirectory for this run's Tensoboard logs.
    if not os.path.exists('./summaries/train/'):
        os.makedirs('./summaries/train/')
    if run_name is None:
        current_dirs = [name for name in os.listdir('./summaries/train/')
                        if os.path.isdir('./summaries/train/' + name)]
        name = model_name + '0'
        count = 0
        while name in current_dirs:
            count += 1
            name = model_name + '{}'.format(count)
        run_name = name

    # Savers and summary writers
    if not os.path.exists('./training'):  # Dir that we'll later save .ckpts to
        os.makedirs('./training')
    if not os.path.exists('./models'):  # Dir that save final models to
        os.makedirs('./models')
    saver = tf.train.Saver()
    final_saver = tf.train.Saver(train_vars)
    merged = tf.summary.merge_all()
    full_log_path = './summaries/train/' + run_name
    train_writer = tf.summary.FileWriter(full_log_path)

    # We must include local variables because of batch pipeline.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # Begin training.
    print ('Starting training...')
    with tf.Session() as sess:
        # Initialization
        sess.run(init_op)
        vggnet.load_weights('libs/vgg16_weights.npz', sess)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                current_step = sess.run(global_step)
                batch = sess.run(batch_op)

                # Collect content targets
                content_data = sess.run(content_layers,
                                        feed_dict={Y: batch})

                feed_dict = {X: batch,
                             content_targets: content_data,
                             beta: beta_val}
                if (current_step % num_steps_ckpt == 0):
                    # Save a checkpoint
                    save_path = 'training/' + model_name + '.ckpt'
                    saver.save(sess, save_path, global_step=global_step)
                    summary, _, loss_out = sess.run([merged, optimizer, loss],
                                                    feed_dict=feed_dict)
                    train_writer.add_summary(summary, current_step)
                    print (current_step, loss_out)

                elif (current_step % 10 == 0):
                    # Collect some diagnostic data for Tensorboard.
                    summary, _, loss_out = sess.run([merged, optimizer, loss],
                                                    feed_dict=feed_dict)
                    train_writer.add_summary(summary, current_step)

                    # Do some standard output.
                    print (current_step, loss_out)
                else:
                    _, loss_out = sess.run([optimizer, loss],
                                           feed_dict=feed_dict)

                # Throw error if we reach number of steps to break after.
                if current_step == num_steps_break:
                    print('Done training.')
                    break
        except tf.errors.OutOfRangeError:
            print('Done training.')
        finally:
            # Save the model (the image transformation network) for later usage
            # in predict.py
            final_saver.save(sess, 'models/' + model_name + '_final.ckpt')

            coord.request_stop()

        coord.join(threads)
    
    
    
def imread(path):
    """Wrapper around cv2.imread. Switches channels to keep everything in RGB.
    :param path:
        String indicating path to image.
    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

    
def imresize(img, scale):
    """Depending on if we scale the image up or down, we use an interpolation
    technique as per OpenCV recommendation.
    :param img:
        3D numpy array of image.
    :param scale:
        float to scale image by in both axes.
    """
    if scale > 1.0:  # use cubic interpolation for upscale.
        img = cv2.resize(img, None, interpolation=cv2.INTER_CUBIC,
                         fx=scale, fy=scale)
    elif scale < 1.0:  # area relation sampling for downscale.
        img = cv2.resize(img, None, interpolation=cv2.INTER_AREA,
                         fx=scale, fy=scale)
    return img
    
    
def imwrite(path, img):
    """Wrapper around cv2.imwrite. Switches it to RGB input convention.
    :param path:
        String indicating path to save image to.
    :param img:
        3D RGB numpy array of image.
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)
    
    
def get_grams(layer_names):
    """Get the style layer tensors from the VGG graph (presumed to be loaded into
    default).
    :param layer_names
        Names of the layers in tf's default graph
    """
    grams = []
    style_layers = get_layers(layer_names)
    for i, layer in enumerate(style_layers):
        b, h, w, c = layer.get_shape().as_list()
        num_elements = h*w*c
        features_matrix = tf.reshape(layer, tf.stack([b, -1, c]))
        gram_matrix = tf.matmul(features_matrix, features_matrix,
                                transpose_a=True)
        gram_matrix = gram_matrix / tf.cast(num_elements, tf.float32)
        grams.append(gram_matrix)
    return grams
    
    
def get_layers(layer_names):
    """Get tensors from default graph by name.
    :param layer_names:
        list of strings corresponding to names of tensors we want to extract.
    """
    g = tf.get_default_graph()
    layers = [g.get_tensor_by_name(name) for name in layer_names]
    return layers
    
def setup_parser():
    """Used to interface with the command-line."""
    parser = argparse.ArgumentParser(
                description='Train a style transfer net.')
    parser.add_argument('--train_dir',
                        default='d:/fast_neural_train_dataset/preprepare',
                        help='Directory of TFRecords training data.')
    parser.add_argument('--model_name',
                        help='Name of model being trained.',
                        default='starsky')
    parser.add_argument('--style_img_path',
                        default='./images/starry_night_crop.jpg',
                        help='Path to style target image.')
    parser.add_argument('--learn_rate',
                        help='Learning rate for Adam optimizer.',
                        default=1e-3, type=float)
    parser.add_argument('--batch_size',
                        help='Batch size for training.',
                        default=4, type=int)
    parser.add_argument('--n_epochs',
                        help='Number of training epochs.',
                        default=2, type=int)
    parser.add_argument('--preprocess_size',
                        help="""Dimensions to resize training images to before passing
                        them into the image transformation network.""",
                        default=[256, 256], nargs=2, type=int)
    parser.add_argument('--run_name',
                        help="""Name of log directory within the Tensoboard
                        directory (./summaries). If not set, will use
                        --model_name to create a unique directory.""",
                        default=None)
    parser.add_argument('--loss_content_layers',
                        help='Names of layers to define content loss.',
                        nargs='*',
                        default=['conv3_3'])
    parser.add_argument('--loss_style_layers',
                        help='Names of layers to define style loss.',
                        nargs='*',
                        default=['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3'])
    parser.add_argument('--content_weights',
                        help="""Weights that multiply the content loss
                        terms.""",
                        nargs='*',
                        default=[1.0],
                        type=float)
    parser.add_argument('--style_weights',
                        help="""Weights that multiply the style loss terms.""",
                        nargs='*',
                        default=[5.0, 5.0, 5.0, 5.0],
                        type=float)
    parser.add_argument('--num_steps_ckpt',
                        help="""Save a checkpoint everytime this number of
                        steps passes in training.""",
                        default=1000,
                        type=int)
    parser.add_argument('--num_pipe_buffer',
                        help="""Number of images loaded into RAM in pipeline.
                        The larger, the better the shuffling, but the more RAM
                        filled, and a slower startup.""",
                        default=4000,
                        type=int)
    parser.add_argument('--num_steps_break',
                        help="""Max on number of steps. Training ends when
                        either num_epochs or this is reached (whichever comes
                        first).""",
                        default=-1,
                        type=int)
    parser.add_argument('--beta',
                        help="""TV regularization weight. If using deconv for
                        --upsample_method, try 1.e-4 for starters. Otherwise,
                        this is not needed.""",
                        default=0.0,
                        type=float)
    parser.add_argument('--style_target_resize',
                        help="""Scale factor to apply to the style target image.
                        Can change the dominant stylistic features.""",
                        default=1.0, type=float)
    parser.add_argument('--upsample_method',
                        help="""Either deconvolution as in the original paper,
                        or the resize convolution method. The latter seems
                        superior and does not require TV regularization through
                        beta.""",
                        choices=['deconv', 'resize'],
                        default='resize')
    return parser
    
    
    
    
    
    
# TODO: For resize-convolution, what if we use strides of 1 for the
# convolution instead of upsampling past the desired dimensions? Test this.

def create_net(X, upsample_method='deconv'):
    """Creates the transformation network, given dimensions acquired from an
    input image. Does this according to J.C. Johnson's specifications
    after utilizing instance normalization (i.e. halving dimensions given
    in the paper).
    :param X
        tf.Tensor with NxHxWxC dimensions.
    :param upsample_method
        values: 'deconv', 'resize'
        Whether to upsample via deconvolution, or the proposed fix of resizing
        + convolution. Description of 2nd method is available at:
            http://distill.pub/2016/deconv_checkerboard/
    """
    assert(upsample_method in ['deconv', 'resize'])

    # Input
    # X = tf.placeholder(tf.float32, shape=shape, name="input")

    # Padding
    h = reflect_pad(X, 40)

    # Initial convolutional layers
    with tf.variable_scope('initconv_0'):
        h = relu(inst_norm(conv2d(h, 3, 16, 9, [1, 1, 1, 1])))
    with tf.variable_scope('initconv_1'):
        h = relu(inst_norm(conv2d(h, 16, 32, 3, [1, 2, 2, 1])))
    with tf.variable_scope('initconv_2'):
        h = relu(inst_norm(conv2d(h, 32, 64, 3, [1, 2, 2, 1])))

    # Residual layers
    with tf.variable_scope('resblock_0'):
        h = res_layer(h, 64, 3, [1, 1, 1, 1])
    with tf.variable_scope('resblock_1'):
        h = res_layer(h, 64, 3, [1, 1, 1, 1])
    with tf.variable_scope('resblock_2'):
        h = res_layer(h, 64, 3, [1, 1, 1, 1])
    with tf.variable_scope('resblock_3'):
        h = res_layer(h, 64, 3, [1, 1, 1, 1])
    with tf.variable_scope('resblock_4'):
        h = res_layer(h, 64, 3, [1, 1, 1, 1])

    # Upsampling layers (tanh on last to get 0,255 range)
    if upsample_method is 'deconv':
        with tf.variable_scope('upsample_0'):
            h = relu(inst_norm(deconv2d(h, 64, 32, 3, [1, 2, 2, 1])))
        with tf.variable_scope('upsample_1'):
            h = relu(inst_norm(deconv2d(h, 32, 16, 3, [1, 2, 2, 1])))
        with tf.variable_scope('upsample_2'):
            h = scaled_tanh(inst_norm(deconv2d(h, 16, 3, 9, [1, 1, 1, 1])))
    elif upsample_method is 'resize':
        with tf.variable_scope('upsample_0'):
            h = relu(inst_norm(upconv2d(h, 64, 32, 3, [1, 2, 2, 1])))
        with tf.variable_scope('upsample_1'):
            h = relu(inst_norm(upconv2d(h, 32, 16, 3, [1, 2, 2, 1])))
        with tf.variable_scope('upsample_2'):  # Not actually an upsample.
            h = scaled_tanh(inst_norm(conv2d(h, 16, 3, 9, [1, 1, 1, 1])))

    # Create a redundant layer with name 'output'
    h = tf.identity(h, name='output')

    return h


def reflect_pad(X, padsize):
    """Pre-net padding.
    :param X
        Input image tensor
    :param padsize
        Amount by which to pad the image tensor
    """
    h = tf.pad(X, paddings=[[0, 0], [padsize, padsize], [padsize, padsize],
                            [0, 0]], mode='REFLECT')
    return h


def conv2d(X, n_ch_in, n_ch_out, kernel_size, strides, name=None,
           padding='SAME'):
    """Creates the convolutional layer.
    :param X
        Input tensor
    :param n_ch_in
        Number of input channels
    :param n_ch_out
        Number of output channels
    :param kernel_size
        Dimension of the square-shaped convolutional kernel
    :param strides
        Length 4 vector of stride information
    :param name
        Optional name for the weight matrix
    """
    if name is None:
        name = 'W'
    shape = [kernel_size, kernel_size, n_ch_in, n_ch_out]
    W = tf.get_variable(name=name,
                        shape=shape,
                        dtype=tf.float32,
                        initializer=tf.random_normal_initializer(stddev=0.1))
    h = tf.nn.conv2d(X,
                     filter=W,
                     strides=strides,
                     padding=padding)
    return h


def upconv2d(X, n_ch_in, n_ch_out, kernel_size, strides):
    """Resizes then applies a convolution.
    :param X
        Input tensor
    :param n_ch_in
        Number of input channels
    :param n_ch_out
        Number of output channels
    :param kernel_size
        Size of square shaped convolutional kernel
    :param strides
        Stride information
    """
    shape = [kernel_size, kernel_size, n_ch_in, n_ch_out]

    # We first upsample two strides-worths. The convolution will then bring it
    # down one stride.
    new_h = X.get_shape().as_list()[1]*strides[1]**2
    new_w = X.get_shape().as_list()[2]*strides[2]**2
    upsized = tf.image.resize_images(X, [new_h, new_w], method=1)

    # Now convolve to get the channels to what we want.
    shape = [kernel_size, kernel_size, n_ch_in, n_ch_out]
    W = tf.get_variable(name='W',
                        shape=shape,
                        dtype=tf.float32,
                        initializer=tf.random_normal_initializer())
    h = tf.nn.conv2d(upsized,
                     filter=W,
                     strides=strides,
                     padding="SAME")

    return h


def deconv2d(X, n_ch_in, n_ch_out, kernel_size, strides):
    """Creates a transposed convolutional (deconvolution) layer.
    :param X
        Input tensor
    :param n_ch_in
        Number of input channels
    :param n_ch_out
        Number of output channels
    :param kernel_size
        Size of square shaped deconvolutional kernel
    :param strides
        Stride information
    """
    # Note the in and out channels reversed for deconv shape
    shape = [kernel_size, kernel_size, n_ch_out, n_ch_in]

    # Construct output shape of the deconvolution
    new_h = X.get_shape().as_list()[1]*strides[1]
    new_w = X.get_shape().as_list()[2]*strides[2]
    output_shape = [X.get_shape().as_list()[0], new_h, new_w, n_ch_out]

    W = tf.get_variable(name='W',
                        shape=shape,
                        dtype=tf.float32,
                        initializer=tf.random_normal_initializer())
    h = tf.nn.conv2d_transpose(X,
                               output_shape=output_shape,
                               filter=W,
                               strides=strides,
                               padding="SAME")

    return h


def relu(X):
    """Performs relu on the tensor.
    :param X
        Input tensor
    """
    return tf.nn.relu(X, name='relu')


def scaled_tanh(X):
    """Performs tanh activation to ensure range of 0,255 on positive output.
    :param X
        Input tensor
    """
    scale = tf.constant(255.0)
    shift = tf.constant(255.0)
    half = tf.constant(2.0)
    # out = tf.mul(tf.tanh(X), scale)  # range of [-255, 255]
    out = (scale*tf.tanh(X) + shift) / half
    # out = tf.add(out, shift)  # range of [0, 2*255]
    # out = tf.div(out, half)  # range of [0, 255]
    return out


def inst_norm(inputs, epsilon=1e-3, suffix=''):
    """
    Assuming TxHxWxC dimensions on the tensor, will normalize over
    the H,W dimensions. Use this before the activation layer.
    This function borrows from:
        http://r2rt.com/implementing-batch-normalization-in-tensorflow.html
    Note this is similar to batch_normalization, which normalizes each
    neuron by looking at its statistics over the batch.
    :param input_:
        input tensor of NHWC format
    """
    # Create scale + shift. Exclude batch dimension.
    stat_shape = inputs.get_shape().as_list()
    scale = tf.get_variable('INscale'+suffix,
                            initializer=tf.ones(stat_shape[3]))
    shift = tf.get_variable('INshift'+suffix,
                            initializer=tf.zeros(stat_shape[3]))

    inst_means, inst_vars = tf.nn.moments(inputs, axes=[1, 2],
                                          keep_dims=True)

    # Normalization
    inputs_normed = (inputs - inst_means) / tf.sqrt(inst_vars + epsilon)

    # Perform trainable shift.
    output = scale * inputs_normed + shift

    return output


def res_layer(X, n_ch, kernel_size, strides):
    """Creates a residual block layer.
    :param X
        Input tensor
    :param n_ch
        Number of input channels
    :param kernel_size
        Size of square shaped convolutional kernel
    :param strides
        Stride information
    """
    h = conv2d(X, n_ch, n_ch, kernel_size, strides, name='W1', padding='VALID')
    h = relu(inst_norm(h, suffix='1'))
    h = conv2d(h, n_ch, n_ch, kernel_size, strides, name='W2', padding='VALID')
    h = inst_norm(h, suffix='2')

    # Crop for skip connection
    in_shape = X.get_shape().as_list()
    begin = [0, 2, 2, 0]
    size = [-1, in_shape[1]-4, in_shape[2]-4, -1]
    X_crop = tf.slice(X, begin=begin, size=size)

    # Residual skip connection
    h = tf.add(h, X_crop, name='res_out')

    return h

    
    
    
    
    
    

def fst_content_loss(content_layers, target_content_layers,
                 content_weights):
    """Defines the content loss function.
    :param content_layers
        List of tensors for layers derived from training graph.
    :param target_content_layers
        List of placeholders to be filled with content layer data.
    :param content_weights
        List of floats to be used as weights for content layers.
    """
    assert(len(target_content_layers) == len(content_layers))
    num_content_layers = len(target_content_layers)

    # Content loss
    content_losses = []
    for i in range(num_content_layers):
        content_layer = content_layers[i]
        target_content_layer = target_content_layers[i]
        content_weight = content_weights[i]
        loss = tf.reduce_sum(tf.squared_difference(content_layer,
                                                   target_content_layer))
        loss = content_weight * loss
        _, h, w, c = content_layer.get_shape().as_list()
        num_elements = h * w * c
        loss = loss / tf.cast(num_elements, tf.float32)
        content_losses.append(loss)
    content_loss = tf.add_n(content_losses, name='content_loss')
    return content_loss


def fst_style_loss(grams, target_grams, style_weights):
    """Defines the style loss function.
    :param grams
        List of tensors for Gram matrices derived from training graph.
    :param target_grams
        List of numpy arrays for Gram matrices precomputed from style image.
    :param style_weights
        List of floats to be used as weights for style layers.
    """
    assert(len(grams) == len(target_grams))
    num_style_layers = len(target_grams)

    # Style loss
    style_losses = []
    for i in range(num_style_layers):
        gram, target_gram = grams[i], target_grams[i]
        style_weight = style_weights[i]
        _, c1, c2 = gram.get_shape().as_list()
        size = c1*c2
        loss = tf.reduce_sum(tf.square(gram - tf.constant(target_gram)))
        loss = style_weight * loss / size
        style_losses.append(loss)
    style_loss = tf.add_n(style_losses, name='style_loss')
    return style_loss


def fst_tv_loss(X):
    """Creates 2d TV loss using X as the input tensor. Acts on different colour
    channels individually, and uses convolution as a means of calculating the
    differences.
    :param X:
        4D Tensor
    """
    # These filters for the convolution will take the differences across the
    # spatial dimensions. Constructing these on paper has to be done carefully,
    # but can be easily understood  when one realizes that the sub-3x3 arrays
    # should have no mixing terms as the RGB channels should not interact
    # within this convolution. Thus, the 2 3x3 subarrays are identity and
    # -1*identity. The filters should look like:
    # v_filter = [ [(3x3)], [(3x3)] ]
    # h_filter = [ [(3x3), (3x3)] ]
    ident = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    v_array = np.array([[ident], [-1*ident]])
    h_array = np.array([[ident, -1*ident]])
    v_filter = tf.constant(v_array, tf.float32)
    h_filter = tf.constant(h_array, tf.float32)

    vdiff = tf.nn.conv2d(X, v_filter, strides=[1, 1, 1, 1], padding='VALID')
    hdiff = tf.nn.conv2d(X, h_filter, strides=[1, 1, 1, 1], padding='VALID')

    loss = tf.reduce_sum(tf.square(hdiff)) + tf.reduce_sum(tf.square(vdiff))

    return loss
    
    
    
    
    
    
    
def preprocessing(image, resize_shape):
    """Simply resizes the image.
    :param image:
        image tensor
    :param resize_shape:
        list of dimensions
    """
    if resize_shape is None:
        return image
    else:
        image = tf.image.resize_images(image, size=resize_shape, method=2)
        return image


def read_my_file_format(filename_queue, resize_shape=None):
    """Sets up part of the pipeline that takes elements from the filename queue
    and turns it into a tf.Tensor of a batch of images.
    :param filename_queue:
        tf.train.string_input_producer object
    :param resize_shape:
        2 element list defining the shape to resize images to.
    """
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example, features={
            'image/encoded': tf.FixedLenFeature([], tf.string),
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/channels': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64)})
    example = tf.image.decode_jpeg(features['image/encoded'], 3)
    processed_example = preprocessing(example, resize_shape)
    return processed_example


def batcher(filenames, batch_size, resize_shape=None, num_epochs=None,
            min_after_dequeue=4000):
    """Creates the batching part of the pipeline.
    :param filenames:
        list of filenames
    :param batch_size:
        size of batches that get output upon each access.
    :param resize_shape:
        for preprocessing. What to resize images to.
    :param num_epochs:
        number of epochs that define end of training set.
    :param min_after_dequeue:
        min_after_dequeue defines how big a buffer we will randomly sample
        from -- bigger means better shuffling but slower start up and more
        memory used.
        capacity must be larger than min_after_dequeue and the amount larger
        determines the maximum we will prefetch.  Recommendation:
        min_after_dequeue + (num_threads + a small safety margin) * batch_size
    """
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs, shuffle=True)
    example = read_my_file_format(filename_queue, resize_shape)
    capacity = min_after_dequeue + 3 * batch_size
    example_batch = tf.train.shuffle_batch(
        [example], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return example_batch

    
    
    
    
    
    
    
    
# set parameters
content_filename = 'images/medN_100.jpg'
style_filename = 'images/starryNight.jpg'
result_img_path = './images/styled.jpg'

weight_content = 1.0
weight_style = 10.0
weight_denoise = 0.3


content_image = load_image(content_filename, max_size=None)
style_image = load_image(style_filename, max_size=300)

model_name = 'Stonehenge'

content_layer_ids = [6]
style_layer_ids = [1, 3, 6, 9]

parser = setup_parser()
args = parser.parse_args()
    
img = style_transfer(args, False,
                     content_image=content_image,
                     style_image=style_image,
                     content_layer_ids=content_layer_ids,
                     style_layer_ids=style_layer_ids,
                     content_img_path=content_filename, 
                     style_img_path=style_filename,
                     result=result_img_path,
                     model_name=model_name,
                     weight_content=weight_content,
                     weight_style=weight_style,
                     weight_denoise=weight_denoise,
                     num_iterations=101,
                     step_size=10.0)
    

