import numpy as np
import tensorflow as tf
import pandas as pd

class LightGCN(tf.keras.Model):

    def __init__(self, hparams, data, seed=None):
        super(LightGCN, self).__init__()
        
        tf.random.set_seed(seed)
        np.random.seed(seed)

        self.data = data
        self.epochs = hparams.epochs
        self.lr = hparams.learning_rate
        self.emb_dim = hparams.embed_size
        self.batch_size = hparams.batch_size
        self.n_layers = hparams.n_layers
        self.decay = hparams.decay
        self.eval_epoch = hparams.eval_epoch
        self.top_k = hparams.top_k
        self.save_model = hparams.save_model
        self.save_epoch = hparams.save_epoch
        self.metrics = hparams.metrics
        self.model_dir = hparams.MODEL_DIR

        self.norm_adj = data.get_norm_adj_mat()
        self.n_users = data.n_users
        self.n_items = data.n_items

        self.users = tf.keras.layers.Input(shape=(), dtype=tf.int32)
        self.pos_items = tf.keras.layers.Input(shape=(), dtype=tf.int32)
        self.neg_items = tf.keras.layers.Input(shape=(), dtype=tf.int32)

        self.weights = self._init_weights()
        self.ua_embeddings, self.ia_embeddings = self._create_lightgcn_embed()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
    
    def _init_weights(self):
        initializer = tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
        return {
            "user_embedding": tf.Variable(initializer([self.n_users, self.emb_dim]), name="user_embedding"),
            "item_embedding": tf.Variable(initializer([self.n_items, self.emb_dim]), name="item_embedding")
        }
    
    def _create_lightgcn_embed(self):
        A_hat = self._convert_sp_mat_to_sp_tensor(self.norm_adj)
        ego_embeddings = tf.concat([self.weights["user_embedding"], self.weights["item_embedding"]], axis=0)
        all_embeddings = [ego_embeddings]

        for _ in range(self.n_layers):
            # embedding propagatin
            ego_embeddings = tf.sparse.sparse_dense_matmul(A_hat, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        
        all_embeddings = tf.reduce_mean(tf.stack(all_embeddings, axis=1), axis=1)
        return tf.split(all_embeddings, [self.n_users, self.n_items], axis=0) # separate user and item embeddings
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32) # convert to coordinate format: [(r,c,v)]
        indices = np.array([coo.row, coo.col]).T
        return tf.sparse.SparseTensor(indices, coo.data, coo.shape)
    
    @tf.function 
    def train_step(self, users, pos_items, neg_items):
        with tf.GradientTape() as tape:
            u_emb = tf.nn.embedding_lookup(self.ua_embeddings, users)
            pos_i_emb = tf.nn.embedding_lookup(self.ia_embeddings, pos_items)
            neg_i_emb = tf.nn.embedding_lookup(self.ia_embeddings, neg_items)
        
            mf_loss, emb_loss = self._create_bpr_loss(u_emb, pos_i_emb, neg_i_emb)
            loss = mf_loss + emb_loss 
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss, mf_loss, emb_loss 

    def _create_bpr_loss(self, user_embeddings, pos_item_embeddings, neg_item_embeddings):
        pos_scores = tf.reduce_sum(user_embeddings * pos_item_embeddings, axis=1)
        neg_scores = tf.reduce_sum(user_embeddings * neg_item_embeddings, axis=1)

        regularizer = tf.nn.l2_loss(self.weights["user_embeddings"]) + tf.nn.l2_loss(self.weights["item_embedding"])
        regularizer = regularizer / self.batch_size

        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
        emb_loss = self.decay * regularizer
        return mf_loss, emb_loss
    
    def fit(self):
        for epoch in range(1, self.epochs + 1):
            loss, mf_loss, emb_loss = 0.0, 0.0, 0.0
            n_batch = self.data.train.shape[0] // self.batch_size + 1

            for _ in range(n_batch):
                users, pos_items, neg_items = self.data.train_loader(self.batch_size)
                batch_loss, batch_mf_loss, batch_emb_loss = self.train_step(users, pos_items, neg_items)
                loss += batch_loss / n_batch 
                mf_loss += batch_mf_loss / n_batch 
                emb_loss += batch_emb_loss / n_batch 

                print(f"Epoch {epoch}: Loss={loss:.5f}, MF_Loss={mf_loss:.5f}, Emb_Loss={emb_loss:.5f}")

                if self.save_model and epoch % self.save_epoch == 0:
                    save_path = os.path.join(self.model_dir, f"epoch_{epoch}")
                    self.save_weights(save_path)
                    print(f"Model saved at {save_path}")
    
    def score(self, user_ids):
        u_emb = tf.nn.embedding_lookup(self.ua_embeddings, user_ids)
        all_item_emb = self.ia_embeddings
        scores = tf.matmul(u_emb, all_item_emb, transpose_b=True)
        return scores.numpy()

    def recommend_k_items(self, user_ids, top_k=10):
        scores = self.score(user_ids)
        top_items = np.argsort(-scores, axis=1)[:, :top_k]
        return top_items

    