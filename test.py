from utils import *
import glob
import model

class Test(object):
    def __init__(self,input_path, output_path, model_path, sigma, conf):
        self.input_path=input_path
        self.output_path=output_path
        self.model_path=model_path
        self.conf=conf
        self.sigma=sigma/255.

    def __call__(self):
        img_list=np.sort(np.asarray(glob.glob('%s/*.png' % self.input_path)))
        print(img_list)

        input=tf.placeholder(tf.float32, shape=[None, None, None, 3])

        EST=model.Estimator(input,'EST')
        sigma_hat=EST.output

        MODEL=model.Denoiser(input,sigma_hat,'Denoise1')
        output=MODEL.output

        saverE=tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='EST'))
        count_param('EST')

        vars1=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Denoise1')
        vars2=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Noise_ENC')
        saverM=tf.train.Saver(var_list=vars1+vars2)

        count_param('Denoise1')
        count_param('Noise_ENC')
        count_param()

        print('Sigma: ', self.sigma*255.)

        with tf.Session(config=self.conf) as sess:
            ckpt_modelE = os.path.join(self.model_path, 'EST')
            print(ckpt_modelE)
            saverE.restore(sess,ckpt_modelE)

            ckpt_model = os.path.join(self.model_path, 'AWGN')
            print(ckpt_model)
            saverM.restore(sess,ckpt_model)

            P = []

            print('Process %d images' % len(img_list))
            for idx, img_path in enumerate(img_list):
                img=imread(img_path)
                img=img[None,:,:,:]

                np.random.seed(0)

                noise_img = img + np.random.standard_normal(img.shape)*self.sigma

                out=sess.run(output, feed_dict={input:noise_img})
                P.append(psnr(img[0]*255., np.clip(np.round(out[0]*255.), 0., 255.)))

                if not os.path.exists('%s/Noise%d' % (self.output_path, self.sigma*255.)):
                    os.makedirs('%s/Noise%d'% (self.output_path, self.sigma*255.))
                imageio.imsave('%s/Noise%d/%s.png'% (self.output_path, self.sigma*255., os.path.basename(img_path[:-4])), np.uint8(np.clip(np.round(out[0] * 255.), 0., 255.)))

                if idx % 5 == 0:
                    print('[%d/%d] Processing' % ((idx+1), len(img_list)))

            print('PSNR: %.4f' % np.mean(P))