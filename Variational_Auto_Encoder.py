import tensorflow as tf
import tensorflow.keras.layers as layers
from Preprocess import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sn

tf.keras.backend.set_floatx('float64')
writer= tf.summary.create_file_writer('VAE\logs_graph')
train_mode= True
test_mode = False
# ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=net, iterator=iterator)
# manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)

class Sampling(layers.Layer):
    @tf.function
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        # with tf.name_scope('z_samples'):
        #     z_samples= z_mean + tf.exp(0.5 * z_log_var) * epsilon
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):

    def __init__(self,
               latent_dim=32,
               intermediate_dim=64,
               name='encoder',
               **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.embed_layer = layers.Embedding(64, 128, name='embeding_layer')
        self.dense_in= layers.Dense(128,kernel_regularizer=tf.keras.regularizers.l1(.5),name='sparse_input')
        self.dense_proj = layers.Dense(intermediate_dim, activation='relu',name='intermediate_layer')
        self.dense_mean = layers.Dense(latent_dim,name='z_mean')
        self.dense_log_var = layers.Dense(latent_dim,activation='relu',name='z_variance')
        self.sampling = Sampling()

    @tf.function
    def call(self, inputs):
        x_embed= self.embed_layer(inputs)
        x_in = self.dense_in(x_embed)
        x_hidd= self.dense_proj(tf.reshape(x_in,shape=(-1,inputs.shape[1]*128)))
        z_mean = self.dense_mean(x_hidd)
        z_log_var = self.dense_log_var(x_hidd)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(layers.Layer):

    def __init__(self,
               original_dim,
               intermediate_dim=64,
               name='decoder',
               **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation='relu',name='input_layer')
        self.dense_output = layers.Dense(original_dim, activation='relu',name='output_layer')

    @tf.function
    def call(self, inputs):
        x_input = self.dense_proj(inputs)
        return self.dense_output(x_input)


class VariationalAutoEncoder(tf.keras.Model):
    def __init__(self,
               original_dim,
               intermediate_dim=64,
               latent_dim=32,
               name='autoencoder',
               **kwargs):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim,
                           intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

    @tf.function
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        # kl_loss = - 0.5 * tf.reduce_mean(
        #     z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        # self.add_loss(kl_loss)
        return reconstructed, z_mean, z_log_var, z

def create_encoder_model():
    in_layer= layers.Input(shape=(27,))
    embed_layer = layers.Embedding(64, 128, name='embeding_layer')(in_layer)
    dense_in = layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l1(.5), name='sparse_input')(embed_layer)
    reshape= tf.keras.layers.Reshape()
    dense_proj = layers.Dense(128, activation='relu', name='intermediate_layer')(dense_in)
    dense_mean = layers.Dense(32, name='z_mean')(dense_proj)
    dense_log_var = layers.Dense(32, activation='relu', name='z_variance')(dense_proj)
    model= tf.keras.Model(inputs=[in_layer],outputs=[dense_mean,dense_log_var])
    return model

model= create_encoder_model()
mypipline=Pipeline([('read_data',ReadData()),
                    (('clean_data'),CleanData('median')),
                    (('cat_to_num'),CatToNum('ordinal'))
                    # ,('Outlier_mitigation',OutlierDetection(threshold=2,name='whisker'))
                    ])

data= mypipline.fit_transform(None)
features_name= data.columns[0:-1].tolist()

x= data.values[:,0:-1]
y= data.values[:,-1]
original_dim = x.shape[1]
x_train = x[np.where(y==1)[0]]
dense_feature= [0,2,3]
scale= MinMaxScaler(feature_range=(0,10)).fit(x_train[:,dense_feature])
x_train[:,dense_feature]= scale.transform(x_train[:,dense_feature])
buffer_size= x_train.shape[0]



vae = VariationalAutoEncoder(original_dim, 128, 32)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
mse_loss_fn = tf.keras.losses.MeanSquaredError()

loss_metric = tf.keras.metrics.Mean()

train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
myencoder = Encoder(32,128)
mydecoder= Decoder(original_dim,128)

tf.summary.trace_on(graph=True, profiler=True)
z_mean, z_log_var, z = myencoder(x_train[0:100,:])
reconsructed= mydecoder(z)
# reconstructed, z_mean, z_log_var, z = vae(x_train[0:100,:])
with writer.as_default():
    tf.summary.trace_export(name="Encoder-Decoder", step=0, profiler_outdir='VAE\logs_graph')


ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=vae)
manager = tf.train.CheckpointManager(ckpt, 'VAE\saved_model', max_to_keep=3)
# Iterate over epochs.
if train_mode:
    epochs = 1
    global_step = 0
    for epoch in range(epochs):
        print('Start of epoch %d' % (epoch,))
        # Iterate over the batches of the dataset.
        for step, x_batch_train in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                reconstructed, z_mean, z_log_var, z = vae(x_batch_train)
                kl_loss = - 0.5 * tf.reduce_mean(
                    z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)

                loss = mse_loss_fn(x_batch_train, reconstructed)
                loss += kl_loss  # Add KLD regularization loss
                ckpt.step.assign_add(1)
                with writer.as_default():
                    tf.summary.scalar('main loss', loss, global_step)
                    tf.summary.scalar('kl loss', kl_loss, global_step)
                    for i in range(32):
                        tf.summary.scalar('mean_latent_variable_{}'.format(i), tf.reduce_mean(z_mean[:, i]),
                                          global_step)
                        tf.summary.scalar('variance_latent_variable{}'.format(i), tf.reduce_mean(z_log_var[:, i]),
                                          global_step)
                        tf.summary.histogram('latent_variable_dist_{}'.format(i), z[:, i], global_step)

            grads = tape.gradient(loss, vae.trainable_weights)
            optimizer.apply_gradients(zip(grads, vae.trainable_weights))
            global_step += 1
            loss_metric(loss)
            if step % 100 == 0:
                save_path = manager.save()
                vae.save_weights('./VAE/manual_save_ignore/variatinal_ckpt')
                print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                print('step %s: mean loss = %s' % (step, loss_metric.result()))

if test_mode:
    # ckpt.restore(manager.latest_checkpoint)
    load_vae= VariationalAutoEncoder(original_dim, 128, 32)
    load_vae.load_weights('./VAE/manual_save/variatinal_ckpt')
    z= tf.random.normal(shape=[2000,32], mean=4,stddev=2,dtype=tf.float64)
    post_samples= load_vae.decoder(z)
    # post_samples_vae = vae.decoder(z)
    fig, axes = plt.subplots(3,2,sharex='none',sharey='none')
    names=['RTD_ST_CD','Tenure','Age']
    i=0
    for name,ax in zip(names,axes):
        sn.distplot(post_samples[:,dense_feature[i]],hist=True,kde=True,ax=ax[0],
                    label='reconstructed_{}'.format(name))
        sn.distplot(x_train[:,dense_feature[i]],hist=True,kde=True,ax=ax[1],
                    label='original_Data_{}'.format(name))
        i+=1
    fig.savefig('./VAE/pos_samples.png')


