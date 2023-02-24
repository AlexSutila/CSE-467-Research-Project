from keras.layers import Input, Dense, BatchNormalization
from keras.regularizers import l1, l2
from keras import Model, backend
from pandas import DataFrame


class variational_autoencoder:

    def __init__(self, df: DataFrame, latent_dim: int, batch_size: int,
                 hidden_nodes: int):

        encoder_input = Input(shape=(df.shape[1],))
        decoder_input = Input(shape=(latent_dim,))

        # Create the corresponding models to represent the encoder and decoder
        encoder_model = self.__create_encoder(encoder_input,
                                              latent_dim,
                                              hidden_nodes)
        decoder_model = self.__create_decoder(decoder_input,
                                              latent_dim,
                                              df.shape[1],
                                              hidden_nodes)
        vae_output = decoder_model(encoder_model(encoder_input))

        self.vae = Model(encoder_input, vae_output)

    def __create_encoder(self, encoder_input: Input, latent_dim: int,
                         hidden_nodes: int):

        batch_normalization_1 = BatchNormalization()(encoder_input)
        hidden_encoder_layer = Dense(hidden_nodes,
                                     activation='relu',
                                     kernel_regularizer=l2(0.01),
                                     bias_regularizer=l1(0.01)
                                     )(batch_normalization_1)
        batch_normalization_2 = BatchNormalization()(hidden_encoder_layer)

        return Model(encoder_input, Dense(latent_dim)(batch_normalization_2))

    def __create_decoder(self, decoder_input: Input, latent_dim: int,
                         df_cols: tuple, hidden_nodes: int) -> Model:

        batch_normalization_1 = BatchNormalization()(decoder_input)
        hidden_decoder_layer = Dense(hidden_nodes,
                                     activation='relu',
                                     kernel_regularizer=l2(0.01),
                                     bias_regularizer=l1(0.01)
                                     )(batch_normalization_1)
        batch_normalization_2 = BatchNormalization()(hidden_decoder_layer)

        return Model(decoder_input, Dense(df_cols,
                                          activation='linear'
                                          )(batch_normalization_2))

    def get(self) -> Model:
        return self.vae
