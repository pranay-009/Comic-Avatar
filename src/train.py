import numpy as np
import os
def train(gen,dis,comb,batch_size,epochs,val_data,train_files,train_gen):
    discrim=[]
    generator_loss=[]
    half_batch=batch_size//2
    batch_step=int(len(train_files)/batch_size)
    val_gen=[]
    for epoch in range(epochs):
        s=0
        for batch in range(batch_step):
            
            x_real,y_real=train_gen.__getitem__(s)
            x_half=x_real[:half_batch]
            y_half=y_real[:half_batch]
            y_fake=gen.predict(x_half)
            
            
            #training half batch with the real images and another half with fake images(model generated images)
            
            d_loss_real=dis.train_on_batch([y_half,x_half],np.ones((half_batch,8,8,1)))#training the discriminator in real images
            d_loss_fake=dis.train_on_batch([y_fake,x_half],np.zeros((half_batch,8,8,1)))#training the discriminator in fake images
            fak_pred=np.ones((batch_size,8,8,1))
            #now we will try out generator 
            #and to fool the discriminator we pass the generated image labels as the real image
            
            gan_loss,_,_=comb.train_on_batch(x_real,[fak_pred,y_real])
            s=s+batch_size
        dis_loss=(d_loss_fake+d_loss_real)/2
        discrim.append(dis_loss)
        generator_loss.append(gan_loss)
        
        if epoch%5==0:
            val=gen.predict(np.expand_dims(val_data,axis=0))
            val_gen.append(val)
            path="./checkpoint/model-"+str(100+epoch)+".ckpt"
            gen.save_weights(path)
            
        print('epoch> %d,dis_loss[%.3f] generator_loss[%.3f]' % (epoch+1, dis_loss, gan_loss)   )
        
    return discrim,generator_loss,val_gen
