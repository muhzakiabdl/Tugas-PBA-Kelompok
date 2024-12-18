import numpy as np

class TextPreprocessor:
    def __init__(self, config):
        self.config = config
        self.embeddings_index = self.load_glove_embeddings(config.GLOVE_PATH)

    def create_embedding_matrix(self, word_to_idx):
        # Pastikan dimensi embedding sesuai dengan ukuran GloVe yang digunakan
        embedding_dim = self.config.EMBEDDING_DIM  # 100 untuk GloVe 100d
        # Inisialisasi embedding_matrix dengan ukuran yang sesuai
        embedding_matrix = np.zeros((len(word_to_idx), embedding_dim))

        for word, idx in word_to_idx.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # Pastikan embedding_vector memiliki dimensi yang sesuai dengan embedding_dim
                embedding_matrix[idx] = embedding_vector
            else:
                # Jika tidak ada embedding, gunakan vector nol (atau bisa juga random)
                embedding_matrix[idx] = np.zeros(embedding_dim)

        return embedding_matrix

    def load_glove_embeddings(self, glove_path):
        embeddings_index = {}
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        return embeddings_index
