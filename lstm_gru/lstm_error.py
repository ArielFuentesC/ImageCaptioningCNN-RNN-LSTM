import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


class ImageCaptionModel(nn.Module):
    def __init__(self, config: dict):
        """
        This is the main module class for the image captioning network
        :param config: dictionary holding neural network configuration
        """
        super(ImageCaptionModel, self).__init__()
        # Store config values as instance variables
        self.vocabulary_size = config['vocabulary_size']
        self.embedding_size = config['embedding_size']
        self.number_of_cnn_features = config['number_of_cnn_features']
        self.hidden_state_sizes = config['hidden_state_sizes']
        self.num_rnn_layers = config['num_rnn_layers']
        self.cell_type = config['cellType']

        # Create the network layers
        self.embedding_layer = nn.Embedding(self.vocabulary_size, self.embedding_size)
        # TODO: The output layer (final layer) is a linear layer. What should be the size (dimensions) of its output?
        #         Replace None with a linear layer with correct output size
        self.output_layer = nn.Linear(self.hidden_state_sizes,
                                      self.vocabulary_size)  # nn.Linear(self.hidden_state_sizes, )
        self.nn_map_size = 512  # The output size for the image features after the processing via self.inputLayer
        # TODO: Check the task description and replace None with the correct input layer
        self.input_layer = nn.Sequential(nn.Dropout(p=0.25),
                                         nn.Linear(self.number_of_cnn_features, self.nnmapsize),
                                         nn.LeakyReLU())

        self.simplified_rnn = False

        if self.simplified_rnn:
            # Simplified one layer RNN is used for task 1 only.
            if self.cell_type != 'RNN':
                raise ValueError('config["cellType"] must be "RNN" when self.simplified_rnn has been set to True.'
                                 'It is ', self.cell_type, 'instead.')

            if self.num_rnn_layers != 1:
                raise ValueError('config["num_rnn_layers"] must be 1 for simplified RNN.'
                                 'It is', self.num_rnn_layers, 'instead.')

            #self.rnn = RNNOneLayerSimplified(input_size=self.embedding_size + self.nn_map_size,
            #                                 hidden_state_size=self.hidden_state_sizes)
        else:
            self.rnn = RNN(input_size=self.embedding_size + self.nn_map_size,
                           hidden_state_size=self.hidden_state_sizes,
                           num_rnn_layers=self.num_rnn_layers,
                           cell_type=self.cell_type)

    def forward(self, cnn_features, x_tokens, is_train: bool, current_hidden_state=None) -> tuple:
        """
        :param cnn_features: Features from the CNN network, shape[batch_size, number_of_cnn_features]
        :param x_tokens: Shape[batch_size, truncated_backprop_length]
        :param is_train: A flag used to select whether or not to use estimated token as input
        :param current_hidden_state: If not None, it should be passed into the rnn module. It's shape should be
                                    [num_rnn_layers, batch_size, hidden_state_sizes].
        :return: logits of shape [batch_size, truncated_backprop_length, vocabulary_size] and new current_hidden_state
                of size [num_rnn_layers, batch_size, hidden_state_sizes]
        """
        # HINT: For task 4, you might need to do self.input_layer(torch.transpose(cnn_features, 1, 2))
        processed_cnn_features = self.input_layer(cnn_features)

        if current_hidden_state is None:
            # TODO: Initialize initial_hidden_state with correct dimensions depending on the cell type.
            # The shape of the hidden state here should be [num_rnn_layers, batch_size, hidden_state_sizes].
            # Remember that each rnn cell needs its own initial state.
            batch_size = cnn_features.shape[0]
            initial_hidden_state = Variable(torch.zeros([self.num_rnn_layers, batch_size, self.hidden_state_sizes]))
        else:
            initial_hidden_state = current_hidden_state

        # Call self.rnn to get the "logits" and the new hidden state
        logits, hidden_state = self.rnn(x_tokens, processed_cnn_features, initial_hidden_state, self.output_layer,
                                        self.embedding_layer, is_train)

        return logits, hidden_state


######################################################################################################################
class RNN(nn.Module):
    def __init__(self, input_size, hidden_state_size, num_rnn_layers, cell_type='LSTM'):
        """
        :param input_size: Size of the embeddings
        :param hidden_state_size: Number of units in the RNN cells (will be equal for all RNN layers)
        :param num_rnn_layers: Number of stacked RNN layers
        :param cell_type: The type cell to use like vanilla RNN, GRU or GRU.
        """
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_state_size = hidden_state_size
        self.num_rnn_layers = num_rnn_layers
        self.cell_type = cell_type

        input_size_list = [LSTMCell(hidden_state_size, input_size)]
        new_input_size = hidden_state_size
        input_size_list.extend([LSTMCell(hidden_state_size, new_input_size) for i in range(num_rnn_layers - 1)])

        self.cells = nn.ModuleList(input_size_list)

    def forward(self, tokens, processed_cnn_features, initial_hidden_state, output_layer: nn.Linear,
                embedding_layer: nn.Embedding, is_train=True) -> tuple:
        """
        :param tokens: Words and chars that are to be used as inputs for the RNN.
                       Shape: [batch_size, truncated_backpropagation_length]
        :param processed_cnn_features: Output of the CNN from the previous module.
        :param initial_hidden_state: The initial hidden state of the RNN.
        :param output_layer: The final layer to be used to produce the output. Uses RNN's final output as input.
                             It is an instance of nn.Linear
        :param embedding_layer: The layer to be used to generate input embeddings for the RNN.
        :param is_train: Boolean variable indicating whether you're training the network or not.
                         If training mode is off then the predicted token should be fed as the input
                         for the next step in the sequence.
        :return: A tuple (logits, final hidden state of the RNN).
                 logits' shape = [batch_size, truncated_backpropagation_length, vocabulary_size]
                 hidden layer's shape = [num_rnn_layers, batch_size, hidden_state_sizes]
        """

        if is_train:
            sequence_length = tokens.shape[1]  # Truncated backpropagation length
        else:
            sequence_length = 40  # Max sequence length to generate

        # Get embeddings for the whole sequence
        embeddings = embedding_layer(input=tokens)  # Should have shape (batch_size, sequence_length, embedding_size)

        logits_sequence = []
        current_hidden_state = initial_hidden_state.to('cuda') # [nun_rnn_layers, batch_size, 2*hidden_state_sizes]
        input_tokens = embeddings[:,0,:]  # Should have shape (batch_size, embedding_size)

        for i in range(sequence_length):
            for j in range(self.num_rnn_layers):
                if j == 0:
                    input_first_layer = torch.cat((input_tokens, processed_cnn_features), dim=1)
                    output = self.cells[0](input_first_layer, current_hidden_state[0,:,:].clone())
                    current_hidden_state[0,:,:] = torch.unsqueeze(output, 0)

                else:
                    # split hidden_state and cell_memory
                    hidden_state, cell_memory = torch.split(current_hidden_state[(j-1),:,:], self.hidden_state_size , dim=1)
                    output = self.cells[j](hidden_state, current_hidden_state[j,:,:].clone())
                    current_hidden_state[j,:,:] = torch.unsqueeze(output, 0)

            h_state, c_memory = torch.split(output, self.hidden_state_size , dim=1)
            logits_i = output_layer(h_state)
            logits_sequence.append(logits_i)
            predictions = torch.argmax(logits_i, dim=1)

            # Get the input tokens for the next step in the sequence
            if i < sequence_length - 1:
                if is_train:
                    input_tokens = embeddings[:, i+1, :]
                else:
                    input_tokens = embedding_layer(predictions)

        logits = torch.stack(logits_sequence, dim=1)  # Convert the sequence of logits to a tensor

        return logits, current_hidden_state


class RNNexp(nn.Module):
    def __init__(self, input_size, hidden_state_size, num_rnn_layers, cell_type='LSTM'):
        """
        :param input_size: Size of the embeddings
        :param hidden_state_size: Number of units in the RNN cells (will be equal for all RNN layers)
        :param num_rnn_layers: Number of stacked RNN layers
        :param cell_type: The type cell to use like vanilla RNN, GRU or GRU.
        """
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_state_size = hidden_state_size
        self.num_rnn_layers = num_rnn_layers
        self.cell_type = cell_type

        # TODO: len(input_size_list) == num_rnn_layers and input_size_list[i] should contain the input size for layer i.
        # This is used to populate self.cells
        input_size_list = [self.input_size]
        for i in range(self.num_rnn_layers - 1):
            input_size_list.append(self.hidden_state_size)

        # TODO: Create a list of type "nn.ModuleList" and populate it with cells of type
        #       "self.cell_type" - depending on the number of RNN layers.

        if self.cell_type == "LSTM":
            self.cells = nn.ModuleList(
                [LSTMCell(hidden_state_size=self.hidden_state_size, input_size=i) for i in input_size_list])

    def forward(self, tokens, processed_cnn_features, initial_hidden_state, output_layer: nn.Linear,
                embedding_layer: nn.Embedding, is_train=True) -> tuple:
        """
        :param tokens: Words and chars that are to be used as inputs for the RNN.
                       Shape: [batch_size, truncated_backpropagation_length]
        :param processed_cnn_features: Output of the CNN from the previous module.
        :param initial_hidden_state: The initial hidden state of the RNN.
        :param output_layer: The final layer to be used to produce the output. Uses RNN's final output as input.
                             It is an instance of nn.Linear
        :param embedding_layer: The layer to be used to generate input embeddings for the RNN.
        :param is_train: Boolean variable indicating whether you're training the network or not.
                         If training mode is off then the predicted token should be fed as the input
                         for the next step in the sequence.

        :return: A tuple (logits, final hidden state of the RNN).
                 logits' shape = [batch_size, truncated_backpropagation_length, vocabulary_size]
                 hidden layer's shape = [num_rnn_layers, batch_size, hidden_state_sizes]
        """
        if is_train:
            sequence_length = tokens.shape[1]  # Truncated backpropagation length
        else:
            sequence_length = 40  # Max sequence length to generate

        # Get embeddings for the whole sequence
        embeddings = embedding_layer(input=tokens)  # Should have shape (batch_size, sequence_length, embedding_size)

        logits_sequence = []
        current_hidden_state = initial_hidden_state
        # TODO: Fetch the first (index 0) embeddings that should go as input to the RNN.
        # Use these tokens in the loop(s) below
        input_tokens = embeddings[:, 0, :]  # Should have shape (batch_size, embedding_size)
        for i in range(sequence_length):
            # TODO:
            # 1. Loop over the RNN layers and provide them with correct input. Inputs depend on the layer
            #    index so input for layer-0 will be different from the input for other layers.
            # 2. Update the hidden cell state for every layer.

            current_hidden_state[0] = self.cells[0](input_tokens, current_hidden_state[0])
            current_hidden_state[1] = self.cells[1](current_hidden_state[0], current_hidden_state[1])


            # 3. If you are at the last layer, then produce logits_i, predictions. Append logits_i to logits_sequence.
            #    See the simplified rnn for the one layer version.
            if self.cell_type == "LSTM":
                logits_i = output_layer(current_hidden_state[-1, :,0:self.hidden_state_size])

            if self.cell_type == "GRU":
                logits_i = output_layer(current_hidden_state[-1, :, :])

            logits_sequence.append(logits_i)
            # Find the next predicted output element
            predictions = torch.argmax(logits_i, dim=1)

            # Get the input tokens for the next step in the sequence
            if i < sequence_length - 1:
                if is_train:
                    input_tokens = embeddings[:, i+1, :]
                else:
                    # TODO: Compute predictions above and use them here by replacing None with the code in comment
                    input_tokens = embedding_layer(predictions)

        logits = torch.stack(logits_sequence, dim=1)  # Convert the sequence of logits to a tensor

        return logits, current_hidden_state




class LSTMCellExp(nn.Module):
    def __init__(self, hidden_state_size: int, input_size: int):
        """
        :param hidden_state_size: Size (number of units/features) in the hidden state of GRU
        :param input_size: Size (number of units/features) of the input to GRU
        """
        super(LSTMCell, self).__init__()
        self.hidden_state_size = hidden_state_size

        # TODO: Initialise weights and biases for the forget gate (weight_f, bias_f), input gate (w_i, b_i),
        #       output gate (w_o, b_o), and hidden state (weight, bias)
        #       self.weight, self.weight_(f, i, o):
        #           A nn.Parameter with shape [HIDDEN_STATE_SIZE + input_size, HIDDEN_STATE_SIZE].
        #           Initialized using variance scaling with zero mean.
        #       self.bias, self.bias_(f, i, o): A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to two.
        #
        #       Tips:
        #           Variance scaling: Var[W] = 1/n
        #       Note: The actual input tensor will have 2 * HIDDEN_STATE_SIZE because it contains both
        #             hidden state and cell's memory

        # Forget gate parameters
        self.weight_f = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(hidden_state_size + input_size, hidden_state_size )))
        self.bias_f = nn.Parameter(torch.zeros((1, hidden_state_size + input_size))) + 2

        # Input gate parameters
        self.weight_i = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(hidden_state_size + input_size, hidden_state_size )))
        self.bias_i = nn.Parameter(torch.zeros((1, hidden_state_size + input_size))) + 2

        # Output gate parameters
        self.weight_o = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(hidden_state_size + input_size, hidden_state_size )))
        self.bias_o = nn.Parameter(torch.zeros((1, hidden_state_size + input_size))) + 2


        # Memory cell parameters
        self.weight = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(hidden_state_size + input_size,  hidden_state_size )))
        self.bias = nn.Parameter(torch.zeros((1, hidden_state_size + input_size))) + 2

    def forward(self, x, hidden_state):
        """
        Implements the forward pass for an GRU unit.
        :param x: A tensor with shape [batch_size, input_size] containing the input for the GRU.
        :param hidden_state: A tensor with shape [batch_size, 2 * HIDDEN_STATE_SIZE] containing the hidden
                             state and the cell memory. The 1st half represents the hidden state and the
                             2nd half represents the cell's memory
        :return: The updated hidden state (including memory) of the GRU cell.
                 Shape: [batch_size, 2 * HIDDEN_STATE_SIZE]
        """
        # TODO: Implement the GRU equations to get the new hidden state, cell memory and return them.
        #       The first half of the returned value must represent the new hidden state and the second half
        #       new cell state.
        # Split hidden state into the previous hidden state and cell memory
        prev_hidden_state_vec  = hidden_state[:, :self.hidden_state_size]
        prev_cell_memory = prev_hidden_state_vec[:, self.hidden_state_size:]
        prev_hidden_state = prev_hidden_state_vec[:, :self.hidden_state_size]


        # Compute the input gate
        input_gate = torch.sigmoid(torch.matmul(x, self.weight_i[:self.hidden_state_size,:]) + self.bias_i[:,:self.hidden_state_size] + \
                     torch.matmul(prev_hidden_state, self.weight_i[self.hidden_state_size:,:]) + self.bias_i[:,self.hidden_state_size:])

        forget_gate = torch.sigmoid(torch.matmul(x, self.weight_f[:self.hidden_state_size,:]) + self.bias_f[:,:self.hidden_state_size] + \
                      torch.matmul(prev_hidden_state, self.weight_f[self.hidden_state_size:,:]) + self.bias_f[:,self.hidden_state_size:])

        #candidate memory
        candidate_memory = torch.tanh(torch.matmul(x, self.weight[:self.hidden_state_size,:]) + self.bias[:,:self.hidden_state_size] + \
                      torch.matmul(prev_hidden_state, self.weight[self.hidden_state_size:,:]) + self.bias[:,self.hidden_state_size:])

        # Compute the updated cell memory
        cell_memory = forget_gate * prev_cell_memory + input_gate * candidate_memory

        # Compute the output gate
        output_gate = torch.sigmoid(torch.matmul(x, self.weight_o[:self.hidden_state_size,:]) + self.bias_o[:,:self.hidden_state_size] + \
                      torch.matmul(prev_hidden_state, self.weight_o[self.hidden_state_size:,:]) + self.bias_o[:,self.hidden_state_size:])

        # Compute the updated hidden state
        new_hidden_state = output_gate * torch.tanh(cell_memory)
        new_hidden_state = torch.cat([new_hidden_state, cell_memory], dim=1)

        # Concatenate the new hidden state and cell memory to return the updated hidden state (including memory)
        return new_hidden_state

class LSTMCelltry(nn.Module):
    def __init__(self, hidden_state_size: int, input_size: int):
        """
        :param hidden_state_size: Size (number of units/features) in the hidden state of GRU
        :param input_size: Size (number of units/features) of the input to GRU
        """
        super(LSTMCell, self).__init__()
        self.hidden_state_size = hidden_state_size

        # TODO: Initialise weights and biases for the forget gate (weight_f, bias_f), input gate (w_i, b_i),
        #       output gate (w_o, b_o), and hidden state (weight, bias)
        #       self.weight, self.weight_(f, i, o):
        #           A nn.Parameter with shape [HIDDEN_STATE_SIZE + input_size, HIDDEN_STATE_SIZE].
        #           Initialized using variance scaling with zero mean.
        #       self.bias, self.bias_(f, i, o): A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to two.
        #
        #       Tips:
        #           Variance scaling: Var[W] = 1/n
        #       Note: The actual input tensor will have 2 * HIDDEN_STATE_SIZE because it contains both
        #             hidden state and cell's memory

        # Forget gate parameters
        self.weight_f = nn.Parameter(torch.randn((hidden_state_size + input_size, hidden_state_size )) *
                                     (1/((hidden_state_size + input_size) * hidden_state_size)))
        self.bias_f = nn.Parameter(torch.zeros((1, hidden_state_size))) + 2

        # Input gate parameters
        self.weight_i = nn.Parameter(torch.randn((hidden_state_size + input_size, hidden_state_size )) *
                                     (1/((hidden_state_size + input_size) * hidden_state_size)))
        self.bias_i = nn.Parameter(torch.zeros((1, hidden_state_size ))) + 2

        # Output gate parameters
        self.weight_o = nn.Parameter(torch.randn((hidden_state_size + input_size, hidden_state_size )) *
                                     (1/((hidden_state_size + input_size) * hidden_state_size)))
        self.bias_o = nn.Parameter(torch.zeros((1, hidden_state_size ))) +2


        # Memory cell parameters
        self.weight = nn.Parameter(torch.randn((hidden_state_size + input_size, hidden_state_size )) *
                                   (1/((hidden_state_size + input_size) * hidden_state_size)))
        self.bias = nn.Parameter(torch.zeros((1, hidden_state_size)))
        

        """
        # Forget gate parameters
        self.weight_f = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(hidden_state_size + input_size, hidden_state_size)))
        self.bias_f = nn.Parameter(torch.zeros((1, hidden_state_size + input_size))) + 2

        # Input gate parameters
        self.weight_i = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(hidden_state_size + input_size, hidden_state_size)))
        self.bias_i = nn.Parameter(torch.zeros((1, hidden_state_size + input_size))) + 2

        # Output gate parameters
        self.weight_o = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(hidden_state_size + input_size, hidden_state_size)))
        self.bias_o = nn.Parameter(torch.zeros((1, hidden_state_size + input_size))) + 2

        # Memory cell parameters
        self.weight = nn.Parameter(
            nn.init.xavier_normal_(torch.empty(hidden_state_size + input_size, hidden_state_size)))
        self.bias = nn.Parameter(torch.zeros((1, hidden_state_size + input_size)))
        """
    def forward(self,x,  hidden_state):
        """
        Implements the forward pass for an GRU unit.
        :param x: A tensor with shape [batch_size, input_size] containing the input for the GRU.
        :param hidden_state: A tensor with shape [batch_size, 2 * HIDDEN_STATE_SIZE] containing the hidden
                             state and the cell memory. The 1st half represents the hidden state and the
                             2nd half represents the cell's memory
        :return: The updated hidden state (including memory) of the GRU cell.
                 Shape: [batch_size, 2 * HIDDEN_STATE_SIZE]
        """
        # TODO: Implement the GRU equations to get the new hidden state, cell memory and return them.
        #       The first half of the returned value must represent the new hidden state and the second half
        #       new cell state.

        # get input, cell and hidden state
        # concat last hidden and x

        inpt = torch.cat((hidden_state[:, :self.hidden_state_size], x), dim = 1)
        #inpt = torch.cat((hidden_state, x), dim=1)
        #inpt = inpt[:, 0:2*self.hidden_state_size]

        # get last cell
        last_cell = hidden_state [:, self.hidden_state_size:]

        # update gates
        input_gate = torch.sigmoid(torch.mm(inpt, self.weight_i) + self.bias_i)
        forget_gate = torch.sigmoid(torch.mm(inpt, self.weight_f) + self.bias_f)
        output_gate = torch.sigmoid(torch.mm(inpt, self.weight_o) + self.bias_o)
        candidate_gate = torch.tanh(torch.mm(inpt, self.weight) + self.bias)

        # update hiddden and cell state
        updated_cell = torch.mul(forget_gate, last_cell) + torch.mul(input_gate, candidate_gate)

        # new hidden
        hidden_out = torch.mul(output_gate, torch.tanh(updated_cell))

        new_hidden_state = torch.cat((hidden_out, updated_cell), dim = 1)

        return new_hidden_state


class LSTMCell(nn.Module):
    def __init__(self, hidden_state_size: int, input_size: int):
        """
        :param hidden_state_size: Size (number of units/features) in the hidden state of GRU
        :param input_size: Size (number of units/features) of the input to GRU
        """
        super(LSTMCell, self).__init__()
        self.hidden_state_size = hidden_state_size
        self.input_size = input_size
        hidden_pluss_input = hidden_state_size + input_size

        # Forget gate parameters
        self.weight_f = torch.nn.Parameter(torch.randn(hidden_pluss_input, hidden_state_size) / np.sqrt(hidden_pluss_input))
        self.bias_f = torch.nn.Parameter(torch.full((1, hidden_state_size), 2.))
        # Input gate parameters
        self.weight_i = torch.nn.Parameter(torch.randn(hidden_pluss_input, hidden_state_size) / np.sqrt(hidden_pluss_input))
        self.bias_i = torch.nn.Parameter(torch.full((1, hidden_state_size), 2.))
        # Output gate parameters
        self.weight_o = torch.nn.Parameter(torch.randn(hidden_pluss_input, hidden_state_size) / np.sqrt(hidden_pluss_input))
        self.bias_o = torch.nn.Parameter(torch.full((1, hidden_state_size), 2.))
        # Memory cell parameters
        self.weight = torch.nn.Parameter(torch.randn(hidden_pluss_input, hidden_state_size) / np.sqrt(hidden_pluss_input))
        self.bias = torch.nn.Parameter(torch.full((1, hidden_state_size), 2.))

    def forward(self, x, hidden_state):
        """
        Implements the forward pass for an GRU unit.
        :param x: A tensor with shape [batch_size, input_size] containing the input for the GRU.
        :param hidden_state: A tensor with shape [batch_size, 2 * HIDDEN_STATE_SIZE] containing the hidden
                             state and the cell memory. The 1st half represents the hidden state and the
                             2nd half represents the cell's memory
        :return: The updated hidden state (including memory) of the GRU cell.
                 Shape: [batch_size, 2 * HIDDEN_STATE_SIZE]
        """

        hidden_state, cell_memory = torch.split(hidden_state, self.hidden_state_size, dim=1)

        i_t = torch.sigmoid(torch.cat((x, hidden_state), 1).mm(self.weight_i) + self.bias_i)
        o_t = torch.sigmoid(torch.cat((x, hidden_state), 1).mm(self.weight_o) + self.bias_o)
        f_t = torch.sigmoid(torch.cat((x, hidden_state), 1).mm(self.weight_f) + self.bias_f)
        c_hat = torch.tanh(torch.cat((x, hidden_state), 1).mm(self.weight) + self.bias)

        c_t = f_t * cell_memory + i_t * c_hat
        h_t = o_t * torch.tanh(c_t)

        new_hidden_state = torch.cat((h_t, c_t), dim=1)
        return new_hidden_state


######################################################################################################################

def loss_fn(logits, y_tokens, y_weights):
    """
    Weighted softmax cross entropy loss.

    Args:
        logits           : Shape[batch_size, truncated_backprop_length, vocabulary_size]
        y_tokens (labels): Shape[batch_size, truncated_backprop_length]
        y_weights         : Shape[batch_size, truncated_backprop_length]. Add contribution to the total loss only
                           from words existing
                           (the sequence lengths may not add up to #*truncated_backprop_length)

    Returns:
        sum_loss: The total cross entropy loss for all words
        mean_loss: The averaged cross entropy loss for all words

    Tips:
        F.cross_entropy
    """
    eps = 1e-7  # Used to avoid division by 0

    logits = logits.view(-1, logits.shape[2])
    y_tokens = y_tokens.view(-1)
    y_weights = y_weights.view(-1)
    losses = F.cross_entropy(input=logits, target=y_tokens, reduction='none')

    sum_loss = (losses * y_weights).sum()
    mean_loss = sum_loss / (y_weights.sum() + eps)

    return sum_loss, mean_loss

# #####################################################################################################################
# if __name__ == '__main__':
#
#     lossDict = {'logits': logits,
#                 'yTokens': yTokens,
#                 'yWeights': yWeights,
#                 'sumLoss': sumLoss,
#                 'meanLoss': meanLoss
#     }
#
#     sumLoss, meanLoss = loss_fn(logits, yTokens, yWeights)
#



#model = LSTMCell(20,5)

#print(model.forward(torch.rand(1, 40), torch.rand(1, 5)).size())